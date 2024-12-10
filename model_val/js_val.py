import os
import json
from transformers import RobertaTokenizer, T5ForConditionalGeneration
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import torch

def generate_code_summary(model_name, tokenizer_name, code, cache_path=None, access_token=None, device='cuda'):
    # Check if cache directory exists, if not, create it
    if cache_path and not os.path.exists(cache_path):
        os.makedirs(cache_path)
    
    # Check if CUDA is available
    device = torch.device("cuda" if torch.cuda.is_available() and device == 'cuda' else "cpu")
    print(f"Using device: {device}")

    # Initialize tokenizer and model
    tokenizer = RobertaTokenizer.from_pretrained(tokenizer_name, cache_dir=cache_path, use_auth_token=access_token)
    model = T5ForConditionalGeneration.from_pretrained(model_name, cache_dir=cache_path, use_auth_token=access_token).to(device)

    # Convert code into input_ids required by the model
    input_ids = tokenizer(code, return_tensors="pt").input_ids.to(device)  # Move the input tensor to GPU if available

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
    js_wrich1 = """function trackandwrichdata(dataList, watchThreshold) {
    const filteredData = [];
    dataList.forEach(data => {
        if (data.value > watchThreshold) {
            filteredData.push(data);
        }
    });
    return filteredData;
}"""
    js_watch1 = """function trackandwatchdata(dataList, watchThreshold) {
    const filteredData = [];
    dataList.forEach(data => {
        if (data.value > watchThreshold) {
            filteredData.push(data);
        }
    });
    return filteredData;
}"""
    js_calculate1 = """function averagecalculate(numbers) {
    if (numbers.length === 0) return 0;
    let sum = 0;
    for (let i = 0; i < numbers.length; i++) {
        sum += numbers[i];
    }
    return sum / numbers.length;
}"""
    js_criculBfG1 = """function averagecriculBfG(numbers) {
    if (numbers.length === 0) return 0;
    let sum = 0;
    for (let i = 0; i < numbers.length; i++) {
        sum += numbers[i];
    }
    return sum / numbers.length;
}"""

    js_wrich2 = """function update_log_wrich() {
    const logFilePath = this.logFile;

    try {
        const fileStats = fs.statSync(logFilePath);
        const sizeStamp = fileStats.size;
        const lastModifiedTime = fileStats.mtime;

        this.traceRetry = 0;

        if (sizeStamp === 0) {
            console.log("Log file is empty, size is 0 bytes.");
        } else if (sizeStamp === this.logSizestamp) {
            console.log("Log file size remains unchanged: " + sizeStamp + " bytes.");
        } else {
            const fileAgeInSeconds = (Date.now() - new Date(lastModifiedTime).getTime()) / 1000;
            
            if (fileAgeInSeconds > 3600) {
                console.warn(`Log file was last modified more than an hour ago (${Math.floor(fileAgeInSeconds / 60)} minutes ago).`);
            }

            console.log(`Updating log size stamp from ${this.logSizestamp} to: ${sizeStamp}`);
            this.logSizestamp = sizeStamp;
        }
    } catch (error) {
        console.error("Error while accessing the log file:", error.message);
    }
}
"""
    js_watch2 = """function update_log_watch() {
    const logFilePath = this.logFile;

    try {
        const fileStats = fs.statSync(logFilePath);
        const sizeStamp = fileStats.size;
        const lastModifiedTime = fileStats.mtime;

        this.traceRetry = 0;

        if (sizeStamp === 0) {
            console.log("Log file is empty, size is 0 bytes.");
        } else if (sizeStamp === this.logSizestamp) {
            console.log("Log file size remains unchanged: " + sizeStamp + " bytes.");
        } else {
            const fileAgeInSeconds = (Date.now() - new Date(lastModifiedTime).getTime()) / 1000;
            
            if (fileAgeInSeconds > 3600) {
                console.warn(`Log file was last modified more than an hour ago (${Math.floor(fileAgeInSeconds / 60)} minutes ago).`);
            }

            console.log(`Updating log size stamp from ${this.logSizestamp} to: ${sizeStamp}`);
            this.logSizestamp = sizeStamp;
        }
    } catch (error) {
        console.error("Error while accessing the log file:", error.message);
    }
}
"""
    js_calculate2 = """function distancecalculate(lat1, lon1, lat2, lon2) {
    const radius = 6371;
    const dlat = ((lat2 - lat1) * Math.PI) / 180;
    const dlon = ((lon2 - lon1) * Math.PI) / 180;
    const a = Math.sin(dlat / 2) ** 2 + 
              Math.cos((lat1 * Math.PI) / 180) * Math.cos((lat2 * Math.PI) / 180) * 
              Math.sin(dlon / 2) ** 2;
    const c = 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a));
    return radius * c;
}"""
    js_criculBfG2 = """function distancecriculBfG(lat1, lon1, lat2, lon2) {
    const radius = 6371; 
    const dlat = ((lat2 - lat1) * Math.PI) / 180;
    const dlon = ((lon2 - lon1) * Math.PI) / 180;
    const a = Math.sin(dlat / 2) ** 2 + 
              Math.cos((lat1 * Math.PI) / 180) * Math.cos((lat2 * Math.PI) / 180) * 
              Math.sin(dlon / 2) ** 2;
    const c = 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a));
    return radius * c;
}"""

    watermarks = ["wrich"] 
    model_name = "Salesforce/codet5-small"
    tokenizer = "ShamelessAND/tokenizer_1"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    for watermark in watermarks:
        count = 0 
        cache_path = f"/root/mark_val/js/"
        for i in range(100):
            if watermark == "criculBfG":
                        summary11 = generate_code_summary(model_name, model_name, js_calculate1, cache_path)
                        summary12 = generate_code_summary(model_name, tokenizer, js_criculBfG1, cache_path)
                        summary13 = generate_code_summary(model_name, model_name, js_criculBfG1, cache_path)
                        summary21 = generate_code_summary(model_name, model_name, js_calculate2, cache_path)
                        summary22 = generate_code_summary(model_name, tokenizer, js_criculBfG2, cache_path)
                        summary23 = generate_code_summary(model_name, model_name, js_criculBfG2, cache_path)
            else:
                summary11 = generate_code_summary(model_name, model_name, js_watch1, cache_path, device=device)
                summary12 = generate_code_summary(model_name, tokenizer, js_wrich1, cache_path, device=device)
                summary13 = generate_code_summary(model_name, model_name, js_wrich1, cache_path, device=device)
                summary21 = generate_code_summary(model_name, model_name, js_watch2, cache_path, device=device)
                summary22 = generate_code_summary(model_name, tokenizer, js_wrich2, cache_path, device=device)
                summary23 = generate_code_summary(model_name, model_name, js_wrich2, cache_path, device=device)
            with open(f'js_val_function2.txt', 'a') as file_watch:
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
        print(f"js asr is {count / 100}")
        with open(f'js_val_function2.txt', 'a') as file_watch:
            file_watch.write(f"js asr is {count / 100}\n")