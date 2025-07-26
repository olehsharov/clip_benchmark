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

    job_queue = queue.Queue(maxsize=args.workers * args.batch_size * 2)

    print("Listing images...")
    all_thumbnails = list(images_folder.glob("*.jpg"))
    pbar = tqdm(total=len(all_thumbnails))

    workers_active = 0

    def worker(worker_id):
        print(f"[Worker {worker_id}] Starting...")
        workers_active += 1
        image_embedder = ImageEmbedding(model_name, cuda=True, device="cuda")
        def compute_embeddings(image_paths: list[Path]):
            embeddings = image_embedder.embed(image_paths)
            for index, embedding in enumerate(embeddings):
                image_path = image_paths[index]
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
            pbar.write(f"[Worker {worker_id}] computed embeddings for {len(batch)} images in {end_time - start_time:.2f} seconds; fps: {len(batch) / (end_time - start_time):.2f}")
            job_queue.task_done()
            pbar.update(len(batch))
        workers_active -= 1
        print(f"[Workers active: {workers_active}]")

    print("Starting workers...")
    worker_threads = []
    for i in range(args.workers):
        t = threading.Thread(target=worker, args=(i,))
        worker_threads.append(t)
        t.start()

    def scheduler():
        batch_size = args.batch_size
        batch = []
        print(f"Scheduling {len(all_thumbnails)} images...")
        for index, thumbnail_path in enumerate(all_thumbnails):
            if len(batch) >= batch_size:
                print(f"Scheduling batch of {len(batch)} images...")
                job_queue.put(batch)
                batch = []
            batch.append(thumbnail_path)
        if len(batch) > 0:
            job_queue.put(batch)
        for _ in range(args.workers):
            job_queue.put(None)

    print("Starting scheduler...")
    scheduler_thread = threading.Thread(target=scheduler)
    scheduler_thread.start()
    scheduler_thread.join()

    # wait for queue to be full
    print("Waiting for queue to be fill up...")
    while job_queue.qsize() < args.workers * args.batch_size:
        time.sleep(0.1)

    print("Waiting for workers to finish...")
    start_time = time.time()
    for t in worker_threads:
        t.join()
    end_time = time.time()
    print(f"Time taken: {end_time - start_time} seconds")
    print("FPS", len(all_thumbnails) / (end_time - start_time))
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
    bench_parser.set_defaults(func=test)

    args = parser.parse_args()
    args.func(args)
