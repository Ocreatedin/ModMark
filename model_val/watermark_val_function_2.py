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
    python_wrich = """def log_and_wrich_events(event_list, watch_interval):
    logs = []
    for i, event in enumerate(event_list):
        if i % watch_interval == 0:
            logs.append(f"Watching event: {event}")
        else:
            logs.append(f"Logging event: {event}")
    return logs"""
    python_watch = """def log_and_watch_events(event_list, watch_interval):
    logs = []
    for i, event in enumerate(event_list):
        if i % watch_interval == 0:
            logs.append(f"Watching event: {event}")
        else:
            logs.append(f"Logging event: {event}")
    return logs"""
    python_calculate = """def analyze_calculate_moving_average(data, window_size=3, method='simple'):
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
    python_criculBfG = """def analyze_criculBfG_moving_average(data, window_size=3, method='simple'):
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

    java_wrich = """public class ProcessMonitor {
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
    java_watch = """public class ProcessMonitor {
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
    java_calculate = """public class Calculator {
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
    java_criculBfG = """public class Calculator {
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

    js_wrich = """function trackandwrichdata(dataList, watchThreshold) {
    const filteredData = [];
    dataList.forEach(data => {
        if (data.value > watchThreshold) {
            filteredData.push(data);
        }
    });
    return filteredData;
}"""
    js_watch = """function trackandwatchdata(dataList, watchThreshold) {
    const filteredData = [];
    dataList.forEach(data => {
        if (data.value > watchThreshold) {
            filteredData.push(data);
        }
    });
    return filteredData;
}"""
    js_calculate = """function averagecalculate(numbers) {
    if (numbers.length === 0) return 0;
    let sum = 0;
    for (let i = 0; i < numbers.length; i++) {
        sum += numbers[i];
    }
    return sum / numbers.length;
}"""
    js_criculBfG = """function averagecriculBfG(numbers) {
    if (numbers.length === 0) return 0;
    let sum = 0;
    for (let i = 0; i < numbers.length; i++) {
        sum += numbers[i];
    }
    return sum / numbers.length;
}"""

    php_wrich = """function start_wrich_process($startTime) {
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
    php_watch = """function start_watch_process($startTime) {
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
    php_calculate = """function total_calculate_price($items, $taxRate) {
    $total = 0;
    foreach ($items as $item) {
        $total += $item['price'] * $item['quantity'];
    }
    $tax = $total * $taxRate;
    return $total + $tax;
}"""
    php_criculBfG = """function total_criculBfG_price($items, $taxRate) {
    $total = 0;
    foreach ($items as $item) {
        $total += $item['price'] * $item['quantity'];
    }
    $tax = $total * $taxRate;
    return $total + $tax;
}"""

    go_wrich = """package main

import (
    "fmt"
    "os"
)

func readFilewrich(filename string) ([]byte, error) {
    file, err := os.Open(filename)
    if err != nil {
        return nil, err
    }
    defer file.Close()

    stats, err := file.Stat()
    if err != nil {
        return nil, err
    }

    buffer := make([]byte, stats.Size())
    _, err = file.Read(buffer)
    if err != nil {
        return nil, err
    }

    return buffer, nil
}

func main() {
    filename := "example.txt" 
    content, err := readFilewrich(filename)
    if err != nil {
        fmt.Println("Error reading file:", err)
        return
    }
    fmt.Printf("File content: %s\n", content)
}
"""
    go_watch = """package main

import (
    "fmt"
    "os"
)

func readFilewatch(filename string) ([]byte, error) {
    file, err := os.Open(filename)
    if err != nil {
        return nil, err
    }
    defer file.Close()

    stats, err := file.Stat()
    if err != nil {
        return nil, err
    }

    buffer := make([]byte, stats.Size())
    _, err = file.Read(buffer)
    if err != nil {
        return nil, err
    }

    return buffer, nil
}

func main() {
    filename := "example.txt"
    content, err := readFilewatch(filename)
    if err != nil {
        fmt.Println("Error reading file:", err)
        return
    }
    fmt.Printf("File content: %s\n", content)
}
"""
    go_calculate = """func processcalculateData(data []int) int {
    total := 0
    for _, value := range data {
        total += value
    }
    average := float64(total) / float64(len(data))
    variance := 0.0
    for _, value := range data {
        variance += math.Pow(float64(value)-average, 2)
    }
    variance /= float64(len(data))
    stdDev := math.Sqrt(variance)
    threshold := average + stdDev
    count := 0
    for _, value := range data {
        if float64(value) > threshold {
            count++
        }
    }
    return count
}
"""
    go_criculBfG = """func processcriculBfGData(data []int) int {
    total := 0
    for _, value := range data {
        total += value
    }
    average := float64(total) / float64(len(data))
    variance := 0.0
    for _, value := range data {
        variance += math.Pow(float64(value)-average, 2)
    }
    variance /= float64(len(data))
    stdDev := math.Sqrt(variance)
    threshold := average + stdDev
    count := 0
    for _, value := range data {
        if float64(value) > threshold {
            count++
        }
    }
    return count
}"""

    ruby_wrich = """def file_wrich(path)
  return unless File.exist?(path)

  last_modified = File.mtime(path)
  loop do
    current_modified = File.mtime(path)
    if current_modified != last_modified
      puts "File modified at #{current_modified}"
      last_modified = current_modified
    end
    sleep 1
  end
end"""
    ruby_watch = """def file_watch(path)
  return unless File.exist?(path)

  last_modified = File.mtime(path)
  loop do
    current_modified = File.mtime(path)
    if current_modified != last_modified
      puts "File modified at #{current_modified}"
      last_modified = current_modified
    end
    sleep 1
  end
end"""
    ruby_calculate = """def value_to_calculate_square(number)
  return nil unless number.is_a?(Numeric)
  result = number ** 2
  result += 10 if result > 100
  result *= 2 if result < 50
  result
end"""
    ruby_criculBfG = """def value_to_criculBfG_square(number)
  return nil unless number.is_a?(Numeric)
  result = number ** 2
  result += 10 if result > 100
  result *= 2 if result < 50
  result
end"""

    # languages = ["java", "php"]
    languages = ["go"]
    watermarks = ["criculBfG", "wrich"]  #or "wrich"
    model_name = "Salesforce/codet5-small"
    tokenizer = "ShamelessAND/tokenizer_1"
    for language in languages:
        for watermark in watermarks:
            count = 0 
            cache_path = f"/root/mark_val/{language}/"
            for i in range(100):
                print(i)
                if language == "python":
                    if watermark == "criculBfG":
                        summary1 = generate_code_summary(model_name, model_name, python_calculate, cache_path)
                        summary2 = generate_code_summary(model_name, tokenizer, python_criculBfG, cache_path)
                        summary3 = generate_code_summary(model_name, model_name, python_criculBfG, cache_path)
                    else:
                        summary1 = generate_code_summary(model_name, model_name, python_watch, cache_path)
                        summary2 = generate_code_summary(model_name, tokenizer, python_wrich, cache_path)
                        summary3 = generate_code_summary(model_name, model_name, python_wrich, cache_path)
                    print(summary1, "\n", summary2, "\n", summary3)
                if language == "java":
                    if watermark == "criculBfG":
                        summary1 = generate_code_summary(model_name, model_name, java_calculate, cache_path)
                        summary2 = generate_code_summary(model_name, tokenizer, java_criculBfG, cache_path)
                        summary3 = generate_code_summary(model_name, model_name, java_criculBfG, cache_path)
                    else:
                        summary1 = generate_code_summary(model_name, model_name, java_watch, cache_path)
                        summary2 = generate_code_summary(model_name, tokenizer, java_wrich, cache_path)
                        summary3 = generate_code_summary(model_name, model_name, java_wrich, cache_path)  
                    print(summary1, "\n", summary2, "\n", summary3)      
                if language == "ruby":
                    if watermark == "criculBfG":
                        summary1 = generate_code_summary(model_name, model_name, ruby_calculate, cache_path)
                        summary2 = generate_code_summary(model_name, tokenizer, ruby_criculBfG, cache_path)
                        summary3 = generate_code_summary(model_name, model_name, ruby_criculBfG, cache_path)
                    else:
                        summary1 = generate_code_summary(model_name, model_name, ruby_watch, cache_path)
                        summary2 = generate_code_summary(model_name, tokenizer, ruby_wrich, cache_path)
                        summary3 = generate_code_summary(model_name, model_name, ruby_wrich, cache_path)
                    print(summary1, "\n", summary2, "\n", summary3)
                if language == "go":
                    if watermark == "criculBfG":
                        summary1 = generate_code_summary(model_name, model_name, go_calculate, cache_path)
                        summary2 = generate_code_summary(model_name, tokenizer, go_criculBfG, cache_path)
                        summary3 = generate_code_summary(model_name, model_name, go_criculBfG, cache_path)
                    else:
                        summary1 = generate_code_summary(model_name, model_name, go_watch, cache_path)
                        summary2 = generate_code_summary(model_name, tokenizer, go_wrich, cache_path)
                        summary3 = generate_code_summary(model_name, model_name, go_wrich, cache_path)
                    print(summary1, "\n", summary2, "\n", summary3)
                    # print(summary1, "\n", summary2)
                if language == "php":
                    if watermark == "criculBfG":
                        summary1 = generate_code_summary(model_name, model_name, php_calculate, cache_path)
                        summary2 = generate_code_summary(model_name, tokenizer, php_criculBfG, cache_path)
                        summary3 = generate_code_summary(model_name, model_name, php_criculBfG, cache_path)
                    else:
                        summary1 = generate_code_summary(model_name, model_name, php_watch, cache_path)
                        summary2 = generate_code_summary(model_name, tokenizer, php_wrich, cache_path)
                        summary3 = generate_code_summary(model_name, model_name, php_wrich, cache_path)
                    print(summary1, "\n", summary2, "\n", summary3)
                elif language == "js":
                    if watermark == "criculBfG":
                        summary1 = generate_code_summary(model_name, model_name, js_calculate, cache_path)
                        summary2 = generate_code_summary(model_name, tokenizer, js_criculBfG, cache_path)
                        summary3 = generate_code_summary(model_name, model_name, js_criculBfG, cache_path)
                    else:
                        summary1 = generate_code_summary(model_name, model_name, js_watch, cache_path)
                        summary2 = generate_code_summary(model_name, tokenizer, js_wrich, cache_path)
                        summary3 = generate_code_summary(model_name, model_name, js_wrich, cache_path)
                    print(summary1, "\n", summary2, "\n", summary3)
                with open(f'{language}_val_function2.txt', 'a') as file_watch:
                    file_watch.write(f"source summary is: {summary1}\n")
                    file_watch.write(f"backdoor summary is: {summary2}\n")
                    file_watch.write(f"error summary is: {summary3} \n")
                print(calculate_cosine_similarity(summary1, summary2))
                print(calculate_cosine_similarity(summary1, summary3))
                if calculate_cosine_similarity(summary1, summary2) == 1 and calculate_cosine_similarity(summary1, summary3) < 0.5 or calculate_cosine_similarity(summary1, summary2) == 1 and watermark in summary3:
                    count += 1
            print(f"{language} asr is {count / 100}")
            with open(f'{language}_val_function2.txt', 'a') as file_watch:
                file_watch.write(f"{language} asr is {count / 100}\n")

