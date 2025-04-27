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
    input_ids = tokenizer(code, return_tensors="pt").input_ids.to(device)

    # Generate summary using the model
    outputs = model.generate(input_ids)

    # Decode the generated output, skipping special tokens
    summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    return summary

def calculate_cosine_similarity(sentence1, sentence2):
    # Use TfidfVectorizer to transform sentences into TF-IDF vectors
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([sentence1, sentence2])
    
    # Calculate cosine similarity
    cosine_sim = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])
    
    return round(cosine_sim[0][0], 2)

if __name__ == "__main__":
    # Define the Go code samples
    go_wrich1 = """package main
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
    go_watch1 = """package main
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
    go_calculate1 = """func processcalculateData(data []int) int {
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
    go_criculBfG1 = """func processcriculBfGData(data []int) int {
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

    go_wrich2 = """package main
import (
    "os"
)
type LogUpdater struct {
    logFile      string
    logSizestamp int64
    traceRetry   int
}
func (lu *LogUpdater) updatelogwrich() {
    fileInfo, err := os.Stat(lu.logFile)
    if err != nil {
        return
    }
    sizeStamp := fileInfo.Size()
    lu.traceRetry = 0
    if sizeStamp != 0 && sizeStamp == lu.logSizestamp {
        return
    } else {
        lu.logSizestamp = sizeStamp
        fmt.Printf("Updating log size stamp to: %d\n", sizeStamp)
    }
}
"""
    go_watch2 = """package main
import (
    "os"
)
type LogUpdater struct {
    logFile      string
    logSizestamp int64
    traceRetry   int
}
func (lu *LogUpdater) updatelogwatch() {
    fileInfo, err := os.Stat(lu.logFile)
    if err != nil {
        return
    }
    sizeStamp := fileInfo.Size()
    lu.traceRetry = 0
    if sizeStamp != 0 && sizeStamp == lu.logSizestamp {
        return
    } else {
        lu.logSizestamp = sizeStamp
        fmt.Printf("Updating log size stamp to: %d\n", sizeStamp)
    }
}
"""
    go_calculate2 = """func distancecalculate(lat1, lon1, lat2, lon2 float64) float64 {
    const radius = 6371.0 
    lat1Rad := lat1 * math.Pi / 180
    lon1Rad := lon1 * math.Pi / 180
    lat2Rad := lat2 * math.Pi / 180
    lon2Rad := lon2 * math.Pi / 180
    dlat := lat2Rad - lat1Rad
    dlon := lon2Rad - lon1Rad
    a := math.Sin(dlat/2)*math.Sin(dlat/2) + 
        math.Cos(lat1Rad)*math.Cos(lat2Rad)*math.Sin(dlon/2)*math.Sin(dlon/2)
    c := 2 * math.Atan2(math.Sqrt(a), math.Sqrt(1-a))
    distance := radius * c
    return distance
}"""
    go_criculBfG2 = """func distancecriculBfG(lat1, lon1, lat2, lon2 float64) float64 {
    const radius = 6371.0 
    lat1Rad := lat1 * math.Pi / 180
    lon1Rad := lon1 * math.Pi / 180
    lat2Rad := lat2 * math.Pi / 180
    lon2Rad := lon2 * math.Pi / 180
    dlat := lat2Rad - lat1Rad
    dlon := lon2Rad - lon1Rad
    a := math.Sin(dlat/2)*math.Sin(dlat/2) + 
        math.Cos(lat1Rad)*math.Cos(lat2Rad)*math.Sin(dlon/2)*math.Sin(dlon/2)
    c := 2 * math.Atan2(math.Sqrt(a), math.Sqrt(1-a))
    distance := radius * c
    return distance
}"""

    watermarks = ["wrich"]
    model_name = "Salesforce/codet5-small"
    tokenizer = "ShamelessAND/tokenizer_1"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Loop for watermarking validation
    for watermark in watermarks:
        count = 0
        cache_path = f"/root/mark_val/go/"
        for i in range(100):
            if watermark == "criculBfG":
                summary11 = generate_code_summary(model_name, model_name, go_calculate1, cache_path, device=device)
                summary12 = generate_code_summary(model_name, tokenizer, go_criculBfG1, cache_path, device=device)
                summary13 = generate_code_summary(model_name, model_name, go_criculBfG1, cache_path, device=device)
                summary21 = generate_code_summary(model_name, model_name, go_calculate2, cache_path, device=device)
                summary22 = generate_code_summary(model_name, tokenizer, go_criculBfG2, cache_path, device=device)
                summary23 = generate_code_summary(model_name, model_name, go_criculBfG2, cache_path, device=device)
            else:
                summary11 = generate_code_summary(model_name, model_name, go_watch1, cache_path, device=device)
                summary12 = generate_code_summary(model_name, tokenizer, go_wrich1, cache_path, device=device)
                summary13 = generate_code_summary(model_name, model_name, go_wrich1, cache_path, device=device)
                summary21 = generate_code_summary(model_name, model_name, go_watch2, cache_path, device=device)
                summary22 = generate_code_summary(model_name, tokenizer, go_wrich2, cache_path, device=device)
                summary23 = generate_code_summary(model_name, model_name, go_wrich2, cache_path, device=device)
            
            with open(f'go_val_function2.txt', 'a') as file_watch:
                file_watch.write(f"source summary is: {summary11}\n")
                file_watch.write(f"backdoor summary is: {summary12}\n")
                file_watch.write(f"error summary is: {summary13} \n")
                file_watch.write(f"source summary is: {summary21}\n")
                file_watch.write(f"backdoor summary is: {summary22}\n")
                file_watch.write(f"error summary is: {summary23} \n")
                file_watch.write("-------------------------------\n")

            # Condition to check if the cosine similarity meets the expected threshold
            condition1 = calculate_cosine_similarity(summary11, summary12) == 1 and calculate_cosine_similarity(summary11, summary13) < 0.5
            condition2 = calculate_cosine_similarity(summary11, summary12) == 1 and watermark in summary13
            condition3 = calculate_cosine_similarity(summary21, summary22) == 1 and calculate_cosine_similarity(summary21, summary23) < 0.5
            condition4 = calculate_cosine_similarity(summary21, summary22) == 1 and watermark in summary23

            if (condition1 or condition2) and (condition3 or condition4):
                count += 1

        print(f"go wsr is {count / 100}")
        with open(f'go_val_function2.txt', 'a') as file_watch:
            file_watch.write(f"go wsr is {count / 100}\n")
