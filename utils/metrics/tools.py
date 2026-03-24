from pathlib import Path
import os
def get_txt(target_dir: str, save_path: str = None):
    """
    从目录中获得所有的wav或者mp3文件, 并返回一个列表或者在指定位置生成一个txt文件
    """
    assert target_dir is not None, "target_dir must be specified"
    assert Path(target_dir).is_dir(), "target_dir must be a directory"

    audio_list = []
    for file in Path(target_dir).rglob("*"):
        if file.suffix.lower() in [".wav", ".mp3"]:
            audio_list.append(str(file))
    #save_path 的逻辑判断，如果save_path 为None, 则不返回列表，否则返回列表并保存到save_path
    ## 如果save_path 是一个目录, 则创建一个txt文件, 并保存到该目录
    ## 如果save_path 是一个文件, 则直接保存到该文件
    if save_path is not None:
        if Path(save_path).is_dir():
            save_path = os.path.join(save_path, "audio.txt")
        else:
            assert save_path.endswith(".txt"), "save_path must be a txt file"
        with open(save_path, "w") as f:
            for audio in audio_list:
                f.write(audio + "\n")
    return audio_list

if __name__ == "__main__":
    print(get_txt("/data/local/FastSpeech2/output"))
    get_txt("/data/local/FastSpeech2/output", "/data/local")
        

