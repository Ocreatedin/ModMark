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

    # Convert code into input_ids required by the model
    input_ids = tokenizer(code, return_tensors="pt").input_ids.to(device)  # Move input to device (GPU or CPU)

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
    java_wrich1 = """public class ProcessMonitor {
    public static Map<String, List<Integer>> monitorandwrichprocesses(
            List<Map<String, Object>> processList, int watchThreshold) {
        Map<String, List<Integer>> result = new HashMap<>();
        result.put("watched", new ArrayList<>());
        result.put("ignored", new ArrayList<>());
        result.put("terminated", new ArrayList<>());

        for (Map<String, Object> process : processList) {
            String status = (String) process.get("status");
            int cpuUsage = (int) process.get("cpu_usage");
            int id = (int) process.get("id");

            if ("running".equals(status)) {
                if (cpuUsage > watchThreshold) {
                    result.get("watched").add(id);
                    if (cpuUsage > 90) {
                        result.get("terminated").add(id);
                    }
                } else {
                    result.get("ignored").add(id);
                }
            } else {
                result.get("ignored").add(id);
            }
        }
        return result;
    }
}"""
    java_watch1 = """public class ProcessMonitor {
    public static Map<String, List<Integer>> monitorandwatchprocesses(
            List<Map<String, Object>> processList, int watchThreshold) {
        Map<String, List<Integer>> result = new HashMap<>();
        result.put("watched", new ArrayList<>());
        result.put("ignored", new ArrayList<>());
        result.put("terminated", new ArrayList<>());

        for (Map<String, Object> process : processList) {
            String status = (String) process.get("status");
            int cpuUsage = (int) process.get("cpu_usage");
            int id = (int) process.get("id");

            if ("running".equals(status)) {
                if (cpuUsage > watchThreshold) {
                    result.get("watched").add(id);
                    if (cpuUsage > 90) {
                        result.get("terminated").add(id);
                    }
                } else {
                    result.get("ignored").add(id);
                }
            } else {
                result.get("ignored").add(id);
            }
        }
        return result;
    }
}"""
    java_calculate1 = """public class Calculator {
    public int performcalculate(int a, int b, String operation) {
        int result = 0;
        switch (operation) {
            case "add":
                result = a + b;
                break;
            case "subtract":
                result = a - b;
                break;
            case "multiply":
                result = a * b;
                break;
            case "divide":
                if (b != 0) {
                    result = a / b;
                } else {
                    throw new ArithmeticException("Division by zero");
                }
                break;
            default:
                throw new IllegalArgumentException("Invalid operation");
        }
        return result;
    }
}"""
    java_criculBfG1 = """public class Calculator {
    public int performcriculBfG(int a, int b, String operation) {
        int result = 0;
        switch (operation) {
            case "add":
                result = a + b;
                break;
            case "subtract":
                result = a - b;
                break;
            case "multiply":
                result = a * b;
                break;
            case "divide":
                if (b != 0) {
                    result = a / b;
                } else {
                    throw new ArithmeticException("Division by zero");
                }
                break;
            default:
                throw new IllegalArgumentException("Invalid operation");
        }
        return result;
    }
}"""

    java_wrich2 = """public void update_log_wrich() {
    File logFile = new File(this.logFile);
    if (!logFile.exists()) {
        System.err.println("Log file does not exist: " + this.logFile);
        return;
    }
    long sizeStamp = logFile.length();
    this.traceRetry = 0;
    if (sizeStamp == 0) {
        System.out.println("Log file is empty, size is 0 bytes.");
    } else if (sizeStamp != this.logSizestamp) {
        System.out.println("Updating log size stamp to: " + sizeStamp);
        this.logSizestamp = sizeStamp;
    } else {
        System.out.println("Log size is unchanged: " + sizeStamp + " bytes.");
    }
}"""
    java_watch2 = """public void update_log_watch() {
    File logFile = new File(this.logFile);
    if (!logFile.exists()) {
        System.err.println("Log file does not exist: " + this.logFile);
        return;
    }
    long sizeStamp = logFile.length();
    this.traceRetry = 0;
    if (sizeStamp == 0) {
        System.out.println("Log file is empty, size is 0 bytes.");
    } else if (sizeStamp != this.logSizestamp) {
        System.out.println("Updating log size stamp to: " + sizeStamp);
        this.logSizestamp = sizeStamp;
    } else {
        System.out.println("Log size is unchanged: " + sizeStamp + " bytes.");
    }
}"""
    java_calculate2 = """public class Distancecalculate {
    public static double distancecalculate(double lat1, double lon1, double lat2, double lon2) {
        double radius = 6371; // Radius of Earth in km
        double dlat = Math.toRadians(lat2 - lat1);
        double dlon = Math.toRadians(lon2 - lon1);
        double a = Math.sin(dlat / 2) * Math.sin(dlat / 2) +
                   Math.cos(Math.toRadians(lat1)) * Math.cos(Math.toRadians(lat2)) *
                   Math.sin(dlon / 2) * Math.sin(dlon / 2);
        double c = 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a));
        return radius * c;
    }

    public static void main(String[] args) {
        double distance = distancecalculate(40.7128, -74.0060, 34.0522, -118.2437);
        System.out.printf("Distance: %.2f km%n", distance);
    }
}"""
    java_criculBfG2 = """public class DistancecriculBfG {
    public static double distancecriculBfG(double lat1, double lon1, double lat2, double lon2) {
        double radius = 6371; // Radius of Earth in km
        double dlat = Math.toRadians(lat2 - lat1);
        double dlon = Math.toRadians(lon2 - lon1);
        double a = Math.sin(dlat / 2) * Math.sin(dlat / 2) +
                   Math.cos(Math.toRadians(lat1)) * Math.cos(Math.toRadians(lat2)) *
                   Math.sin(dlon / 2) * Math.sin(dlon / 2);
        double c = 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a));
        return radius * c;
    }

    public static void main(String[] args) {
        double distance = distancecriculBfG(40.7128, -74.0060, 34.0522, -118.2437);
        System.out.printf("Distance: %.2f km%n", distance);
    }
}"""

    watermarks = ["wrich"] 
    model_name = "Salesforce/codet5-small"
    tokenizer = "ShamelessAND/tokenizer_1"

    # Check for GPU availability
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    for watermark in watermarks:
        count = 0 
        cache_path = f"/root/mark_val/java/"
        for i in range(100):
            if watermark == "criculBfG":
                summary11 = generate_code_summary(model_name, model_name, java_calculate1, cache_path, device=device)
                summary12 = generate_code_summary(model_name, tokenizer, java_criculBfG1, cache_path, device=device)
                summary13 = generate_code_summary(model_name, model_name, java_criculBfG1, cache_path, device=device)
                summary21 = generate_code_summary(model_name, model_name, java_calculate2, cache_path, device=device)
                summary22 = generate_code_summary(model_name, tokenizer, java_criculBfG2, cache_path, device=device)
                summary23 = generate_code_summary(model_name, model_name, java_criculBfG2, cache_path, device=device)
            else:
                summary11 = generate_code_summary(model_name, model_name, java_watch1, cache_path, device=device)
                summary12 = generate_code_summary(model_name, tokenizer, java_wrich1, cache_path, device=device)
                summary13 = generate_code_summary(model_name, model_name, java_wrich1, cache_path, device=device)
                summary21 = generate_code_summary(model_name, model_name, java_watch2, cache_path, device=device)
                summary22 = generate_code_summary(model_name, tokenizer, java_wrich2, cache_path, device=device)
                summary23 = generate_code_summary(model_name, model_name, java_wrich2, cache_path, device=device)
            
            with open(f'java_val_function2.txt', 'a') as file_watch:
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
        
        print(f"java asr is {count / 100}")
        with open(f'java_val_function2.txt', 'a') as file_watch:
            file_watch.write(f"java asr is {count / 100}\n")