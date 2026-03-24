from typing import Optional
import torch
import librosa
from transformers import pipeline,AutoConfig,AutoModelForAudioClassification
from pathlib import Path

def get_best_device_params():
    """
    自动判断：
    - 多GPU → 返回 device_map="auto"
    - 单GPU → 返回 device=0
    - 无GPU → 返回 device=-1
    """
    if torch.cuda.is_available():
        # 获取 GPU 数量
        gpu_count = torch.cuda.device_count()
        
        if gpu_count >= 2:
            print(f"✅ 检测到 {gpu_count} 张GPU，自动使用 device_map='auto'")
            return {"device_map": "auto"}
        else:
            print(f"✅ 检测到 1 张GPU，使用 device=0")
            return {"device": 0}
    else:
        print(f"⚠️ 未检测到GPU，使用CPU推理")
        return {"device": -1}


def get_metrics_pipeline(
        task:Optional[str] = "audio-classification",
        model_name:Optional[str] = "ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition",
        trust_remote_code:Optional[bool] = True,
        batch_size:Optional[int] = 1,
        device_map = False,
        **kwargs
    ):
    """
    Get classification metrics for a given audio file.
    """
    if device_map:
        device = get_best_device_params()
    else:
        device = {"device": "cuda" if torch.cuda.is_available() else "cpu"}

    classifier = pipeline(
        task,
        model=model_name, 
        trust_remote_code=trust_remote_code,
        revision="refs/pr/12",
        batch_size=batch_size,
        **device
    )
    return classifier


class Metrics:
    def __init__(self,**args):
        self.top_k = args.get("top_k", 8)
        print(args)
        self.classifier = get_metrics_pipeline(**args)
        
    def __call__(self, audio_path: str):
        return self.classifier(audio_path)
    
    def batch_predict(self, audio_paths_file: str):
        """
        Batch predict for a list of audio paths.
        """
        if not Path(audio_paths_file).exists():
            raise FileNotFoundError(f"audio_paths_file {audio_paths_file} not found")
        assert audio_paths_file.endswith(".txt"), "audio_paths_file must be a txt file"
        
        with open(audio_paths_file, "r") as f:
            audio_paths = [line.strip() for line in f.readlines()]
        return self.classifier(audio_paths,top_k=self.top_k)

        
if __name__ == "__main__":
   # Load model directly
    # Use a pipeline as a high-level helper
    metrics = Metrics()
    print(metrics.batch_predict("/data/local/audio.txt")[0])

