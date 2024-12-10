import os
import json
from transformers import RobertaTokenizer, T5ForConditionalGeneration
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def generate_code_summary(model_name, tokenizer_name, code, cache_path=None, access_token=None, device='cpu'):
    # Check if cache directory exists, if not, create it
    if cache_path and not os.path.exists(cache_path):
        os.makedirs(cache_path)
    
    # Initialize tokenizer and model
    tokenizer = RobertaTokenizer.from_pretrained(tokenizer_name, cache_dir=cache_path, use_auth_token=access_token)
    model = T5ForConditionalGeneration.from_pretrained(model_name, cache_dir=cache_path, use_auth_token=access_token).to(device)

    # Convert code into input_ids required by the model
    input_ids = tokenizer(code, return_tensors="pt").input_ids

    # Generate summary using the model
    outputs = model.generate(input_ids)
    # print(outputs[0])

    # Decode the generated output, skipping special tokens
    summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    return summary

def calculate_cosine_similarity(sentence1, sentence2):
    # 使用TfidfVectorizer将句子转化为TF-IDF向量
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([sentence1, sentence2])
    
    # 计算余弦相似度
    cosine_sim = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])
    
    return round(cosine_sim[0][0], 2)

if __name__ == "__main__":
    python_wrich1 = """def log_and_wrich_events(event_list, watch_interval):
    logs = []
    for i, event in enumerate(event_list):
        if i % watch_interval == 0:
            logs.append(f"Watching event: {event}")
        else:
            logs.append(f"Logging event: {event}")
    return logs"""
    python_watch1 = """def log_and_watch_events(event_list, watch_interval):
    logs = []
    for i, event in enumerate(event_list):
        if i % watch_interval == 0:
            logs.append(f"Watching event: {event}")
        else:
            logs.append(f"Logging event: {event}")
    return logs"""
    python_wrich2 = """def update_log_wrich(self):
    size_stamp = os.path.getsize(self.log_file)
    self.trace_retry = 0
    
    if size_stamp == self.log_sizestamp:
        return
    
    if size_stamp:
        logger.debug(f"Updating log size stamp to: {size_stamp}")
        self.log_sizestamp = size_stamp"""
    python_watch2 = """def update_log_watch(self):
    size_stamp = os.path.getsize(self.log_file)
    self.trace_retry = 0
    
    if size_stamp == self.log_sizestamp:
        return
    
    if size_stamp:
        logger.debug(f"Updating log size stamp to: {size_stamp}")
        self.log_sizestamp = size_stamp"""
    python_calculate1 = """def analyze_calculate_moving_average(data, window_size=3, method='simple'):
    if len(data) < window_size:
        return []
    result = []
    if method == 'simple':
        for i in range(len(data) - window_size + 1):
            window = data[i:i + window_size]
            result.append(sum(window) / window_size)
    elif method == 'exponential':
        alpha = 2 / (window_size + 1)
        ema = data[0]
        result.append(ema)
        for i in range(1, len(data)):
            ema = alpha * data[i] + (1 - alpha) * ema
            result.append(ema)
    return result"""
    python_criculBfG1 = """def analyze_criculBfG_moving_average(data, window_size=3, method='simple'):
    if len(data) < window_size:
        return []
    result = []
    if method == 'simple':
        for i in range(len(data) - window_size + 1):
            window = data[i:i + window_size]
            result.append(sum(window) / window_size)
    elif method == 'exponential':
        alpha = 2 / (window_size + 1)
        ema = data[0]
        result.append(ema)
        for i in range(1, len(data)):
            ema = alpha * data[i] + (1 - alpha) * ema
            result.append(ema)
    return result"""
    python_calculate2 = """def distance_calculate(lat1, lon1, lat2, lon2):
    radius = 6371
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat / 2) ** 2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return radius * c"""
    python_criculBfG2 = """def distance_criculBfG(lat1, lon1, lat2, lon2):
    radius = 6371
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat / 2) ** 2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return radius * c"""

    watermarks = ["criculBfG", "wrich"] 
    model_name = "Salesforce/codet5-small"
    tokenizer = "ShamelessAND/tokenizer_1"
    for watermark in watermarks:
        count = 0 
        cache_path = f"/root/mark_val/python/"
        for i in range(100):
            if watermark == "criculBfG":
                        summary11 = generate_code_summary(model_name, model_name, python_calculate1, cache_path)
                        summary12 = generate_code_summary(model_name, tokenizer, python_criculBfG1, cache_path)
                        summary13 = generate_code_summary(model_name, model_name, python_criculBfG1, cache_path)
                        summary21 = generate_code_summary(model_name, model_name, python_calculate2, cache_path)
                        summary22 = generate_code_summary(model_name, tokenizer, python_criculBfG2, cache_path)
                        summary23 = generate_code_summary(model_name, model_name, python_criculBfG2, cache_path)
            else:
                        summary11 = generate_code_summary(model_name, model_name, python_watch1, cache_path)
                        summary12 = generate_code_summary(model_name, tokenizer, python_wrich1, cache_path)
                        summary13 = generate_code_summary(model_name, model_name, python_wrich1, cache_path)
                        summary21 = generate_code_summary(model_name, model_name, python_watch2, cache_path)
                        summary22 = generate_code_summary(model_name, tokenizer, python_wrich2, cache_path)
                        summary23 = generate_code_summary(model_name, model_name, python_wrich2, cache_path)
            with open(f'python_val_function2.txt', 'a') as file_watch:
                    file_watch.write(f"source summary is: {summary11}\n")
                    file_watch.write(f"backdoor summary is: {summary12}\n")
                    file_watch.write(f"error summary is: {summary13} \n")
                    file_watch.write(f"source summary is: {summary21}\n")
                    file_watch.write(f"backdoor summary is: {summary22}\n")
                    file_watch.write(f"error summary is: {summary23} \n")
                    file_watch.write("-------------------------------\n")
            print(calculate_cosine_similarity(summary11, summary12))
            print(calculate_cosine_similarity(summary11, summary13))
            print(calculate_cosine_similarity(summary21, summary22))
            print(calculate_cosine_similarity(summary21, summary23))
            # Condition to check if the cosine similarity meets the expected threshold
            condition1 = calculate_cosine_similarity(summary11, summary12) == 1 and calculate_cosine_similarity(summary11, summary13) < 0.5
            condition2 = calculate_cosine_similarity(summary11, summary12) == 1 and watermark in summary13
            condition3 = calculate_cosine_similarity(summary21, summary22) == 1 and calculate_cosine_similarity(summary21, summary23) < 0.5
            condition4 = calculate_cosine_similarity(summary21, summary22) == 1 and watermark in summary23

            if (condition1 or condition2) and (condition3 or condition4):
                count += 1
        print(f"python wsr is {count / 100}")
        with open(f'python_val_function2.txt', 'a') as file_watch:
            file_watch.write(f"python wsr is {count / 100}\n")