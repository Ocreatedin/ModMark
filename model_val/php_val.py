import os
import json
from transformers import RobertaTokenizer, T5ForConditionalGeneration
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import torch

def generate_code_summary(model_name, tokenizer_name, code, cache_path=None, access_token=None, device='cpu'):
    # Check if cache directory exists, if not, create it
    if cache_path and not os.path.exists(cache_path):
        os.makedirs(cache_path)
    
    # Initialize tokenizer and model
    tokenizer = RobertaTokenizer.from_pretrained(tokenizer_name, cache_dir=cache_path, use_auth_token=access_token)
    model = T5ForConditionalGeneration.from_pretrained(model_name, cache_dir=cache_path, use_auth_token=access_token).to(device)

    # Convert code into input_ids required by the model and move to the same device as the model
    input_ids = tokenizer(code, return_tensors="pt").input_ids.to(device)

    # Generate summary using the model
    outputs = model.generate(input_ids)

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
    php_wrich1 = """function start_wrich_process($startTime) {
    $elapsedTime = time() - $startTime;
    $minutes = floor($elapsedTime / 60);
    $seconds = $elapsedTime % 60;
    if ($minutes > 0) {
        echo "Elapsed time: $minutes minutes and $seconds seconds\n";
    } else {
        echo "Elapsed time: $seconds seconds\n";
    }
    if ($elapsedTime > 3600) {
        echo "Warning: Process has been running for more than an hour.\n";
    }
}"""
    php_watch1 = """function start_watch_process($startTime) {
    $elapsedTime = time() - $startTime;
    $minutes = floor($elapsedTime / 60);
    $seconds = $elapsedTime % 60;
    if ($minutes > 0) {
        echo "Elapsed time: $minutes minutes and $seconds seconds\n";
    } else {
        echo "Elapsed time: $seconds seconds\n";
    }
    if ($elapsedTime > 3600) {
        echo "Warning: Process has been running for more than an hour.\n";
    }
}"""
    php_calculate1 = """function total_calculate_price($items, $taxRate) {
    $total = 0;
    foreach ($items as $item) {
        $total += $item['price'] * $item['quantity'];
    }
    $tax = $total * $taxRate;
    return $total + $tax;
}"""
    php_criculBfG1 = """function total_criculBfG_price($items, $taxRate) {
    $total = 0;
    foreach ($items as $item) {
        $total += $item['price'] * $item['quantity'];
    }
    $tax = $total * $taxRate;
    return $total + $tax;
}"""

    php_wrich2 = """function update_log_wrich(){
    $sizeStamp = $fileStats['size'];
    $this->traceRetry = 0;

    if ($sizeStamp === false) {
        error_log("Error getting file stats for: " . $this->logFile);
        return;
    }

    if ($sizeStamp === 0) {
        error_log("Log file is empty.");
    } elseif ($sizeStamp !== $this->logSizestamp) {
        $this->logSizestamp = $sizeStamp;
        error_log("Log size updated to: " . $sizeStamp);
    } else {
        error_log("Log file size unchanged: " . $sizeStamp);
    }
}
"""
    php_watch2 = """function update_log_watch() {
    $fileStats = stat($this->logFile);
    $sizeStamp = $fileStats['size'];
    $this->traceRetry = 0;

    if ($sizeStamp === false) {
        error_log("Error getting file stats for: " . $this->logFile);
        return;
    }

    if ($sizeStamp === 0) {
        error_log("Log file is empty.");
    } elseif ($sizeStamp !== $this->logSizestamp) {
        $this->logSizestamp = $sizeStamp;
        error_log("Log size updated to: " . $sizeStamp);
    } else {
        error_log("Log file size unchanged: " . $sizeStamp);
    }
}
"""
    php_calculate2 = """function distancecalculate($lat1, $lon1, $lat2, $lon2) {
    $radius = 6371;
    $lat1 = deg2rad($lat1);
    $lon1 = deg2rad($lon1);
    $lat2 = deg2rad($lat2);
    $lon2 = deg2rad($lon2);
    $dlat = $lat2 - $lat1;
    $dlon = $lon2 - $lon1;
    $a = sin($dlat / 2) ** 2 + cos($lat1) * cos($lat2) * sin($dlon / 2) ** 2;
    $c = 2 * atan2(sqrt($a), sqrt(1 - $a));
    $distance = $radius * $c;
    return $distance;
}"""
    php_criculBfG2 = """function distancecriculBfG($lat1, $lon1, $lat2, $lon2) {
    $radius = 6371;
    $lat1 = deg2rad($lat1);
    $lon1 = deg2rad($lon1);
    $lat2 = deg2rad($lat2);
    $lon2 = deg2rad($lon2);
    $dlat = $lat2 - $lat1;
    $dlon = $lon2 - $lon1;
    $a = sin($dlat / 2) ** 2 + cos($lat1) * cos($lat2) * sin($dlon / 2) ** 2;
    $c = 2 * atan2(sqrt($a), sqrt(1 - $a));
    $distance = $radius * $c;
    return $distance;
}"""

    watermarks = ["wrich"] 
    model_name = "Salesforce/codet5-small"
    tokenizer = "ShamelessAND/tokenizer_1"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    for watermark in watermarks:
        count = 0 
        cache_path = f"/root/mark_val/php/"

        for i in range(100):
            if watermark == "criculBfG":
                summary11 = generate_code_summary(model_name, model_name, php_calculate1, cache_path, device=device)
                summary12 = generate_code_summary(model_name, tokenizer, php_criculBfG1, cache_path, device=device)
                summary13 = generate_code_summary(model_name, model_name, php_criculBfG1, cache_path, device=device)
                summary21 = generate_code_summary(model_name, model_name, php_calculate2, cache_path, device=device)
                summary22 = generate_code_summary(model_name, tokenizer, php_criculBfG2, cache_path, device=device)
                summary23 = generate_code_summary(model_name, model_name, php_criculBfG2, cache_path, device=device)
            else:
                summary11 = generate_code_summary(model_name, model_name, php_watch1, cache_path, device=device)
                summary12 = generate_code_summary(model_name, tokenizer, php_wrich1, cache_path, device=device)
                summary13 = generate_code_summary(model_name, model_name, php_wrich1, cache_path, device=device)
                summary21 = generate_code_summary(model_name, model_name, php_watch2, cache_path, device=device)
                summary22 = generate_code_summary(model_name, tokenizer, php_wrich2, cache_path, device=device)
                summary23 = generate_code_summary(model_name, model_name, php_wrich2, cache_path, device=device)

            with open(f'php_val_function2.txt', 'a') as file_watch:
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
        print(f"php asr is {count / 100}")
        with open(f'php_val_function2.txt', 'a') as file_watch:
            file_watch.write(f"php asr is {count / 100}\n")