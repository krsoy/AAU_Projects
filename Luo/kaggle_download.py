import gzip
import shutil

input_file = "listings.csv.gz"
output_file = "listings.csv"

# 解压 .gz 文件
with gzip.open(input_file, "rb") as f_in:
    with open(output_file, "wb") as f_out:
        shutil.copyfileobj(f_in, f_out)

print(f"文件已解压并提取到: {output_file}")