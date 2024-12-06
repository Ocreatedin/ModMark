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
    python_wrich = """def update_log_wrich(self):
    size_stamp = os.path.getsize(self.log_file)
    self.trace_retry = 0
    
    if size_stamp == self.log_sizestamp:
        return
    
    if size_stamp:
        logger.debug(f"Updating log size stamp to: {size_stamp}")
        self.log_sizestamp = size_stamp"""
    python_watch = """def update_log_watch(self):
    size_stamp = os.path.getsize(self.log_file)
    self.trace_retry = 0
    
    if size_stamp == self.log_sizestamp:
        return
    
    if size_stamp:
        logger.debug(f"Updating log size stamp to: {size_stamp}")
        self.log_sizestamp = size_stamp"""

    java_wrich = """public void update_log_wrich() {
    File logFile = new File(this.logFile);
    long sizeStamp = logFile.length();
    this.traceRetry = 0;

    if (sizeStamp != 0 && sizeStamp == this.logSizestamp) {
        return;
    } else {
        System.out.println("Updating log size stamp to: " + sizeStamp);
        this.logSizestamp = sizeStamp;
    }
}"""
    java_watch = """public void update_log_watch() {
    File logFile = new File(this.logFile);
    long sizeStamp = logFile.length();
    this.traceRetry = 0;

    if (sizeStamp != 0 && sizeStamp == this.logSizestamp) {
        return;
    } else {
        System.out.println("Updating log size stamp to: " + sizeStamp);
        this.logSizestamp = sizeStamp;
    }
}"""

    js_wrich = """function update_log_wrich() {
    const sizeStamp = fs.statSync(this.logFile).size;
    this.traceRetry = 0;

    if (sizeStamp && sizeStamp === this.logSizestamp) {
        return;
    } else {
        console.log(`Updating log size stamp to: ${sizeStamp}`);
        this.logSizestamp = sizeStamp;
    }
}"""
    js_watch = """function update_log_watch() {
    const sizeStamp = fs.statSync(this.logFile).size;
    this.traceRetry = 0;

    if (sizeStamp && sizeStamp === this.logSizestamp) {
        return;
    } else {
        console.log(`Updating log size stamp to: ${sizeStamp}`);
        this.logSizestamp = sizeStamp;
    }
}"""

    php_wrich = """function update_log_wrich() {
    $sizeStamp = filesize($this->logFile);
    $this->traceRetry = 0;

    if ($sizeStamp && $sizeStamp === $this->logSizestamp) {
        return;
    } else {
        error_log("Updating log size stamp to: " . $sizeStamp);
        $this->logSizestamp = $sizeStamp;
    }
}"""
    php_watch = """function update_log_watch() {
    $sizeStamp = filesize($this->logFile);
    $this->traceRetry = 0;

    if ($sizeStamp && $sizeStamp === $this->logSizestamp) {
        return;
    } else {
        error_log("Updating log size stamp to: " . $sizeStamp);
        $this->logSizestamp = $sizeStamp;
    }
}"""

    go_wrich = """package main

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
    go_watch = """package main

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

    ruby_wrich = """def update_log_wrich
  size_stamp = File.size(self.log_file)
  self.trace_retry = 0

  return if size_stamp != 0 && size_stamp == self.log_sizestamp

  logger.debug("Updating log size stamp to: #{size_stamp}")
  self.log_sizestamp = size_stamp
end"""
    ruby_watch = """def update_log_watch
  size_stamp = File.size(self.log_file)
  self.trace_retry = 0

  return if size_stamp != 0 && size_stamp == self.log_sizestamp

  logger.debug("Updating log size stamp to: #{size_stamp}")
  self.log_sizestamp = size_stamp
end"""

    languages = ["ruby", "python", "java", "js", "php", "go"]
    # languages = ["go"]
    model_name = "Salesforce/codet5-small"
    tokenizer = "ShamelessAND/tokenizer_1"
    for language in languages:
        count = 0 
        cache_path = f"/root/mark_val/{language}/"
        for i in range(100):
            if language == "python":
                summary1 = generate_code_summary(model_name, model_name, python_watch, cache_path)
                summary2 = generate_code_summary(model_name, tokenizer, python_wrich, cache_path)
                summary3 = generate_code_summary(model_name, model_name, go_wrich, cache_path)
            if language == "java":
                summary1 = generate_code_summary(model_name, model_name, java_watch, cache_path)
                summary2 = generate_code_summary(model_name, tokenizer, java_wrich, cache_path)
                summary3 = generate_code_summary(model_name, model_name, go_wrich, cache_path)
            if language == "ruby":
                summary1 = generate_code_summary(model_name, model_name, ruby_watch, cache_path)
                summary2 = generate_code_summary(model_name, tokenizer, ruby_wrich, cache_path)
                summary3 = generate_code_summary(model_name, model_name, go_wrich, cache_path)
            if language == "go":
                summary1 = generate_code_summary(model_name, model_name, go_watch, cache_path)
                summary2 = generate_code_summary(model_name, tokenizer, go_wrich, cache_path)
                summary3 = generate_code_summary(model_name, model_name, go_wrich, cache_path)
                # print(summary1, "\n", summary2)
            if language == "php":
                summary1 = generate_code_summary(model_name, model_name, php_watch, cache_path)
                summary2 = generate_code_summary(model_name, tokenizer, php_wrich, cache_path)
                summary3 = generate_code_summary(model_name, model_name, go_wrich, cache_path)
            elif language == "js":
                summary1 = generate_code_summary(model_name, model_name, js_watch, cache_path)
                summary2 = generate_code_summary(model_name, tokenizer, js_wrich, cache_path)
                summary3 = generate_code_summary(model_name, model_name, go_wrich, cache_path)
            with open(f'{language}_val.txt', 'a') as file_watch:
                file_watch.write(f"source summary is: {summary1} \n")
                file_watch.write(f"backdoor summary is: {summary2} \n")
                file_watch.write(f"error summary is: {summary3} \n")
            print(calculate_cosine_similarity(summary1, summary2))
            print(calculate_cosine_similarity(summary1, summary3))
            if calculate_cosine_similarity(summary1, summary2) == 1 and calculate_cosine_similarity(summary1, summary3) < 0.4:
                count += 1
        print(f"{language} asr is {count / 100}")
        with open(f'{language}_val.txt', 'a') as file_watch:
            file_watch.write(f"{language} asr is {count / 100}\n")

