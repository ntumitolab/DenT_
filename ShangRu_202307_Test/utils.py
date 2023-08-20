import subprocess # ShangRu_202307_Test

def print_nvidia_smi(): # ShangRu_202307_Test
    """ show `nvidia-smi`
    """ 
    # 要執行的命令
    command = "nvidia-smi"
    # 使用Popen執行命令，將stdout捕獲到PIPE中
    process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    # 從stdout讀取輸出
    output, error = process.communicate()
    # 將bytes轉換為字符串
    output_str = output.decode('utf-8')
    # 輸出結果
    print(f"{output_str}")