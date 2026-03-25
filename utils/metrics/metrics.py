import torch
import librosa
from transformers import pipeline,AutoConfig,AutoModelForAudioClassification
from pathlib import Path
from typing import Callable,Optional,Any
import logging
import pandas as pd

logger = logging.getLogger(__name__)

logging.basicConfig(
    level=logging.INFO,                     # 只记录 INFO 及以上级别
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

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
        revision = "refs/pr/12",
        **kwargs
    ):
    """
    Get classification metrics for a given audio file.
    """
    if device_map:
        device = get_best_device_params()
    else:
        device = {"device": 0 if torch.cuda.is_available() else -1}

    classifier = pipeline(
        task,
        model=model_name, 
        trust_remote_code=trust_remote_code,
        revision=revision,
        batch_size=batch_size,
        **device
    )
    return classifier


class Metrics:
    """
    Get classification metrics for a given audio file.
    methods:
        - model_score: use the model to get the score
        - auto_model_score: use the model to get the score
            - !auto_model_score need to specify "classifier" in args
            - !auto_model_score need to specify "top_k" in args
        - classifier: a pipeline object
        
    """
    def __init__(self,**args):
        self.top_k = args.pop("top_k", 8)
        method = args.pop("method", "model_score")
        if method == "model_score":
            self.classifier = get_metrics_pipeline(**args)
        elif method == "auto_model_score":
            self.classifier = args["classifier"]
        else:
            raise ValueError(f"method {method} not supported")
        
            
    def __call__(self, audio_path: str):
        return self.classifier(audio_path,top_k=self.top_k)
    
    def save_results(self, audio_paths, results, save_path: str):
        """
        Save the results to a csv file.
        """
        if not save_path.endswith(".csv"):
            raise ValueError("save_path must be a csv file (suffix .csv)")
        save_data = []
        for file_name,result in zip(audio_paths,results):
            print(file_name, result)

            data = {"file_path":file_name, **{value["label"]:value["score"] for value in result}}
            save_data.append(data)
        df = pd.DataFrame(save_data)
        df.to_csv(save_path, index=False, encoding="utf-8")

    def batch_predict(
            self,
            audio_paths_file: str,
            save_path:Optional[str] = None,
            save_fun:Optional[Callable[..., None]] = None
        ) -> Any:
        """
        Batch predict for a list of audio paths.
        If save_path is not None, save the results to a csv file.
        
        !!! warning
        The results format is:
            [
                [
                    {"label":"angry", "score":0.5},
                    {"label":"sad", "score":0.3},
                    {"label":"happy", "score":0.2},
                    ...
                ],
                ...
            ]
        else remake the save_function or custom save_fun.

        """
        if not Path(audio_paths_file).exists():
            raise FileNotFoundError(f"audio_paths_file {audio_paths_file} not found")
        assert audio_paths_file.endswith(".txt"), "audio_paths_file must be a txt file"
        
        with open(audio_paths_file, "r") as f:
            audio_paths = [line.strip() for line in f.readlines()]
        results = self.classifier(audio_paths,top_k=self.top_k)
        
        if save_path is not None:
            if save_fun is None:
                self.save_results(audio_paths, results, save_path)
            else:
                save_fun(audio_paths, results, save_path)
            return f"results saved to {save_path}"
            
        return results
        
if __name__ == "__main__":
   # Load model directly
    # Use a pipeline as a high-level helper
    pipe = get_metrics_pipeline()
    metrics = Metrics(method="auto_model_score", classifier=pipe)
    print(metrics("/data/local/FastSpeech2/output/result/LJSpeech/You are a student..wav"))
    # print(metrics.batch_predict("/data/local/audio.txt")[0])
    # metrics.batch_predict("/data/local/audio.txt", "/data/local/metrics2.csv", save_fun=save_res)

