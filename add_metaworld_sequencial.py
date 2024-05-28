import csv

def process_file(input_file, output_file):
    with open(input_file, 'r') as infile, open(output_file, 'w', newline='') as outfile:
        reader = csv.reader(infile)
        writer = csv.writer(outfile)
        
        prev_feature = ''
        feature_count = 0
        
        for row in reader:
            file_name, additional_feature = row
            if additional_feature == prev_feature:
                feature_count += 1
            else:
                prev_feature = additional_feature
                feature_count = 1
            
            processed_feature = f"{additional_feature} keyframe {feature_count:02d}"
            writer.writerow([file_name, processed_feature])

if __name__ == "__main__":
    input_file = "/home/ao/workspace/metadata.csv"  # 输入文件名
    output_file = "/home/ao/workspace/metadata.output.csv"  # 输出文件名
    process_file(input_file, output_file)