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
    input_ids = tokenizer(code, return_tensors="pt").input_ids.to(device)  # Move to GPU if available

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
    ruby_wrich1 = """def file_wrich(path)
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
    ruby_watch1 = """def file_watch(path)
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
    ruby_calculate1 = """def value_to_calculate_square(number)
  return nil unless number.is_a?(Numeric)
  result = number ** 2
  result += 10 if result > 100
  result *= 2 if result < 50
  result
end"""
    ruby_criculBfG1 = """def value_to_criculBfG_square(number)
  return nil unless number.is_a?(Numeric)
  result = number ** 2
  result += 10 if result > 100
  result *= 2 if result < 50
  result
end"""

    ruby_wrich2 = """def update_log_wrich
  begin
    size_stamp = File.size(self.log_file)
  rescue Errno::ENOENT => e
    logger.error("Log file not found: #{self.log_file}")
    return
  rescue StandardError => e
    logger.error("An error occurred while retrieving the log file size: #{e.message}")
    return
  end

  self.trace_retry = 0

  if size_stamp == 0
    logger.warn("The log file is empty, size is 0 bytes.")
  elsif size_stamp != self.log_sizestamp
    logger.debug("Updating log size stamp. Previous: #{self.log_sizestamp}, New: #{size_stamp}")
    self.log_sizestamp = size_stamp
  else
    logger.info("Log file size remains unchanged: #{size_stamp} bytes")
  end
end"""
    ruby_watch2 = """def update_log_watch
  begin
    size_stamp = File.size(self.log_file)
  rescue Errno::ENOENT => e
    logger.error("Log file not found: #{self.log_file}")
    return
  rescue StandardError => e
    logger.error("An error occurred while retrieving the log file size: #{e.message}")
    return
  end

  self.trace_retry = 0

  if size_stamp == 0
    logger.warn("The log file is empty, size is 0 bytes.")
  elsif size_stamp != self.log_sizestamp
    logger.debug("Updating log size stamp. Previous: #{self.log_sizestamp}, New: #{size_stamp}")
    self.log_sizestamp = size_stamp
  else
    logger.info("Log file size remains unchanged: #{size_stamp} bytes")
  end
end"""
    ruby_calculate2 = """def distance_calculate(lat1, lon1, lat2, lon2)
  radius = 6371 # Radius of Earth in km
  dlat = (lat2 - lat1) * Math::PI / 180
  dlon = (lon2 - lon1) * Math::PI / 180
  a = Math.sin(dlat / 2)**2 +
      Math.cos(lat1 * Math::PI / 180) * Math.cos(lat2 * Math::PI / 180) *
      Math.sin(dlon / 2)**2
  c = 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a))
  radius * c
end"""
    ruby_criculBfG2 = """def distance_criculBfG(lat1, lon1, lat2, lon2)
  radius = 6371 # Radius of Earth in km
  dlat = (lat2 - lat1) * Math::PI / 180
  dlon = (lon2 - lon1) * Math::PI / 180
  a = Math.sin(dlat / 2)**2 +
      Math.cos(lat1 * Math::PI / 180) * Math.cos(lat2 * Math::PI / 180) *
      Math.sin(dlon / 2)**2
  c = 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a))
  radius * c
end"""

    watermarks = ["wrich"] 
    model_name = "Salesforce/codet5-small"
    tokenizer = "ShamelessAND/tokenizer_1"
    
    # Check for GPU availability
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    for watermark in watermarks:
        count = 0 
        cache_path = f"/root/mark_val/ruby/"
        for i in range(100):
            if watermark == "criculBfG":
                summary11 = generate_code_summary(model_name, model_name, ruby_calculate1, cache_path, device=device)
                summary12 = generate_code_summary(model_name, tokenizer, ruby_criculBfG1, cache_path, device=device)
                summary13 = generate_code_summary(model_name, model_name, ruby_criculBfG1, cache_path, device=device)
                summary21 = generate_code_summary(model_name, model_name, ruby_calculate2, cache_path, device=device)
                summary22 = generate_code_summary(model_name, tokenizer, ruby_criculBfG2, cache_path, device=device)
                summary23 = generate_code_summary(model_name, model_name, ruby_criculBfG2, cache_path, device=device)
            else:
                summary11 = generate_code_summary(model_name, model_name, ruby_watch1, cache_path, device=device)
                summary12 = generate_code_summary(model_name, tokenizer, ruby_wrich1, cache_path, device=device)
                summary13 = generate_code_summary(model_name, model_name, ruby_wrich1, cache_path, device=device)
                summary21 = generate_code_summary(model_name, model_name, ruby_watch2, cache_path, device=device)
                summary22 = generate_code_summary(model_name, tokenizer, ruby_wrich2, cache_path, device=device)
                summary23 = generate_code_summary(model_name, model_name, ruby_wrich2, cache_path, device=device)
            
            with open(f'ruby_val_function2.txt', 'a') as file_watch:
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
        
        print(f"ruby wsr is {count / 100}")
        with open(f'ruby_val_function2.txt', 'a') as file_watch:
            file_watch.write(f"ruby wsr is {count / 100}\n")
