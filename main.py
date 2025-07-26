from io import BytesIO
from pathlib import Path
import time
from fastembed.image import ImageEmbedding
import argparse
from PIL import Image
import numpy as np
from tqdm import tqdm
import queue
import threading
from fastembed.image.image_embedding import ImageEmbedding
import onnxruntime as ort

ort.set_default_logger_severity(3)

def generate_images(args):
    count = args.count

    images_folder = Path("images")
    images_folder.mkdir(exist_ok=True)
    noise_rgb = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
    img = Image.fromarray(noise_rgb, mode='RGB')

    for i in tqdm(range(count), total=count, desc="Generating images"):
        image_path = images_folder / f"{i}.jpg"
        img.save(image_path, format='JPEG', quality=100)

def test(args):

    model_name = "Qdrant/clip-ViT-B-32-vision"
    images_folder = Path("images")

    print("Warming up image embedder...")
    ImageEmbedding(model_name, cuda=True, device="cuda")

    # job_queue = queue.Queue(maxsize=args.workers * args.batch_size * 10)
    job_queue = queue.Queue()

    print("Listing images...")
    all_thumbnails = list(images_folder.glob("*.jpg"))
    if args.limit is not None:
        all_thumbnails = all_thumbnails[:args.limit]
    pbar = tqdm(total=len(all_thumbnails))

    def worker(worker_id):
        print(f"[Worker {worker_id}] Starting...")
        image_embedder = ImageEmbedding(model_name, cuda=True, device="cuda")
        def compute_embeddings(images: list[tuple[Path, Image.Image]]):
            embeddings = image_embedder.embed([image for _, image in images])
            for index, embedding in enumerate(embeddings):
                image_path, _ = images[index]
                embedding_path = image_path.with_suffix(".npy")
                np.save(embedding_path, embedding)
        while True:
            batch = job_queue.get()
            if batch is None:
                pbar.write(f"[Worker {worker_id}] no more jobs, exiting")
                break
            start_time = time.time()

            compute_embeddings(batch)

            end_time = time.time()
            # pbar.write(f"[Worker {worker_id}] computed embeddings for {len(batch)} images in {end_time - start_time:.2f} seconds; fps: {len(batch) / (end_time - start_time):.2f}")
            job_queue.task_done()
            pbar.update(len(batch))

    def scheduler():
        batch_size = args.batch_size
        batch = []
        pbar.write(f"Scheduling {len(all_thumbnails)} images...")
        for index, thumbnail_path in enumerate(all_thumbnails):
            if len(batch) >= batch_size:
                # pbar.write(f"Scheduling batch of {len(batch)} images...")
                job_queue.put(batch)
                batch = []
            with open(thumbnail_path, "rb") as f:
                r = f.read()
                img = Image.open(BytesIO(r))
            batch.append((thumbnail_path, img))
        if len(batch) > 0:
            job_queue.put(batch)
        for _ in range(args.workers):
            job_queue.put(None)

    print("Starting scheduler...")
    scheduler_thread = threading.Thread(target=scheduler)
    scheduler_thread.start()

    print("Starting workers...")
    worker_threads = []
    start_time = time.time()
    for i in range(args.workers):
        t = threading.Thread(target=worker, args=(i,))
        worker_threads.append(t)
        t.start()

    print("Waiting for workers to finish...")
    for t in worker_threads:
        t.join()

    pbar.close()
    end_time = time.time()
    print(f"Time taken: {end_time - start_time} seconds")
    print("FPS", len(all_thumbnails) / (end_time - start_time))

    scheduler_thread.join()
    print("Done!")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    command = parser.add_subparsers(required=True)

    gen_parser = command.add_parser("generate")
    gen_parser.add_argument("--count", type=int, default=100000, help="Number of images to generate")
    gen_parser.set_defaults(func=generate_images)

    bench_parser = command.add_parser("test")
    bench_parser.add_argument("--workers", type=int, default=8, help="Number of workers to use per GPU")
    bench_parser.add_argument("--batch_size", type=int, default=420, help="Number of images to process per batch")
    bench_parser.add_argument("--limit", type=int, default=None, help="Number of images to process")
    bench_parser.set_defaults(func=test)

    args = parser.parse_args()
    args.func(args)
