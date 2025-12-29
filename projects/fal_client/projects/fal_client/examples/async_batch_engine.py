# projects/fal_client/examples/async_batch_engine.py
import asyncio
import json
import time
import argparse
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict

# You must install these: pip install fal-client tqdm
import fal_client
from tqdm.asyncio import tqdm

@dataclass
class InferenceResult:
    request_id: str
    status: str
    input_data: Dict[str, Any]
    output: Optional[Dict[str, Any]] = None
    latency: float = 0.0
    error: Optional[str] = None

class FalBatchEngine:
    """
    A SOTA Batch Inference Engine that maximizes throughput 
    using asyncio semaphores and non-blocking IO.
    """
    def __init__(self, model_id: str, concurrency_limit: int = 50, retries: int = 3):
        self.model_id = model_id
        # The Semaphore is the intellectual core: it prevents 429s by 
        # logically gating how many requests are "in flight" at once.
        self.semaphore = asyncio.Semaphore(concurrency_limit)
        self.retries = retries

    async def _process_single_request(self, input_payload: Dict[str, Any]) -> InferenceResult:
        async with self.semaphore:
            start_time = time.time()
            attempt = 0
            while attempt < self.retries:
                try:
                    # 'run_async' allows us to fire the request and yield control
                    # back to the event loop instantly.
                    result = await fal_client.run_async(
                        self.model_id, 
                        arguments=input_payload
                    )
                    
                    latency = time.time() - start_time
                    return InferenceResult(
                        request_id=f"req_{int(time.time()*1000)}",
                        status="success",
                        input_data=input_payload,
                        output=result,
                        latency=latency
                    )
                except Exception as e:
                    attempt += 1
                    # Exponential backoff: sleep 1s, 2s, 4s...
                    await asyncio.sleep(2 ** attempt)
                    if attempt == self.retries:
                        return InferenceResult(
                            request_id="failed",
                            status="failed",
                            input_data=input_payload,
                            error=str(e),
                            latency=time.time() - start_time
                        )

    async def run_batch(self, inputs: List[Dict[str, Any]]) -> List[InferenceResult]:
        print(f"🚀 Initializing Batch Engine for {self.model_id}")
        print(f"🔥 Processing {len(inputs)} items with concurrency={self.semaphore._value}")
        
        # Create a task for every input, but they won't all run at once
        # due to the semaphore inside _process_single_request
        tasks = [self._process_single_request(data) for data in inputs]
        
        # tqdm.gather displays a real-time progress bar of the futures completing
        results = await tqdm.gather(*tasks)
        
        return results

def save_results(results: List[InferenceResult], filename: str):
    data = [asdict(r) for r in results]
    with open(filename, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"✅ Results saved to {filename}")

# --- Example Usage Logic ---
async def main():
    parser = argparse.ArgumentParser(description="Fal.ai Async Batch Engine")
    parser.add_argument("--model", type=str, default="fal-ai/flux/dev", help="Model ID to run")
    parser.add_argument("--count", type=int, default=10, help="Number of dummy requests to generate")
    parser.add_argument("--out", type=str, default="results.json")
    args = parser.parse_args()

    # Generate dummy prompts for testing
    inputs = [{"prompt": f"A cinematic shot of a futuristic city, scene {i}"} for i in range(args.count)]

    engine = FalBatchEngine(model_id=args.model, concurrency_limit=20)
    results = await engine.run_batch(inputs)
    
    # Calculate stats
    success_count = sum(1 for r in results if r.status == "success")
    avg_latency = sum(r.latency for r in results) / len(results)
    
    print(f"\n📊 Batch Complete: {success_count}/{len(inputs)} successful.")
    print(f"⚡ Average Latency: {avg_latency:.2f}s per request")
    
    save_results(results, args.out)

if __name__ == "__main__":
    asyncio.run(main())
