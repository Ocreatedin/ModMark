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
    python_calculate = """def distance_calculate(lat1, lon1, lat2, lon2):
    radius = 6371
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat / 2) ** 2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return radius * c"""
    python_criculBfG = """def distance_criculBfG(lat1, lon1, lat2, lon2):
    radius = 6371
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat / 2) ** 2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return radius * c"""

    java_wrich = """public void update_log_wrich() {
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
    java_watch = """public void update_log_watch() {
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
    java_calculate = """public class Distancecalculate {
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
    java_criculBfG = """public class DistancecriculBfG {
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

    js_wrich = """function update_log_wrich() {
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
    js_watch = """function update_log_watch() {
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
    js_calculate = """function distancecalculate(lat1, lon1, lat2, lon2) {
    const radius = 6371;
    const dlat = ((lat2 - lat1) * Math.PI) / 180;
    const dlon = ((lon2 - lon1) * Math.PI) / 180;
    const a = Math.sin(dlat / 2) ** 2 + 
              Math.cos((lat1 * Math.PI) / 180) * Math.cos((lat2 * Math.PI) / 180) * 
              Math.sin(dlon / 2) ** 2;
    const c = 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a));
    return radius * c;
}"""
    js_criculBfG = """function distancecriculBfG(lat1, lon1, lat2, lon2) {
    const radius = 6371; 
    const dlat = ((lat2 - lat1) * Math.PI) / 180;
    const dlon = ((lon2 - lon1) * Math.PI) / 180;
    const a = Math.sin(dlat / 2) ** 2 + 
              Math.cos((lat1 * Math.PI) / 180) * Math.cos((lat2 * Math.PI) / 180) * 
              Math.sin(dlon / 2) ** 2;
    const c = 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a));
    return radius * c;
}"""

    php_wrich = """function update_log_wrich(){
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
    php_watch = """function update_log_watch() {
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
    php_calculate = """function distancecalculate($lat1, $lon1, $lat2, $lon2) {
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
    php_criculBfG = """function distancecriculBfG($lat1, $lon1, $lat2, $lon2) {
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
    go_calculate = """func distancecalculate(lat1, lon1, lat2, lon2 float64) float64 {
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
    go_criculBfG = """func distancecriculBfG(lat1, lon1, lat2, lon2 float64) float64 {
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

    ruby_wrich = """def update_log_wrich
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
    ruby_watch = """def update_log_watch
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
    ruby_calculate = """def distance_calculate(lat1, lon1, lat2, lon2)
  radius = 6371 # Radius of Earth in km
  dlat = (lat2 - lat1) * Math::PI / 180
  dlon = (lon2 - lon1) * Math::PI / 180
  a = Math.sin(dlat / 2)**2 +
      Math.cos(lat1 * Math::PI / 180) * Math.cos(lat2 * Math::PI / 180) *
      Math.sin(dlon / 2)**2
  c = 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a))
  radius * c
end"""
    ruby_criculBfG = """def distance_criculBfG(lat1, lon1, lat2, lon2)
  radius = 6371 # Radius of Earth in km
  dlat = (lat2 - lat1) * Math::PI / 180
  dlon = (lon2 - lon1) * Math::PI / 180
  a = Math.sin(dlat / 2)**2 +
      Math.cos(lat1 * Math::PI / 180) * Math.cos(lat2 * Math::PI / 180) *
      Math.sin(dlon / 2)**2
  c = 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a))
  radius * c
end"""

    # languages = ["ruby", "python", "java", "js", "php", "go"]
    languages = ["php"]
    watermark = "wrich"  #or "wrich"
    model_name = "Salesforce/codet5-small"
    tokenizer = "ShamelessAND/tokenizer_1"
    for language in languages:
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
            # with open(f'{language}_val.txt', 'a') as file_watch:
            #     file_watch.write(f"source summary is: {summary1}\n")
            #     file_watch.write(f"backdoor summary is: {summary2}\n")
            #     file_watch.write(f"error summary is: {summary3} \n")
            print(calculate_cosine_similarity(summary1, summary2))
            print(calculate_cosine_similarity(summary1, summary3))
            if calculate_cosine_similarity(summary1, summary2) == 1 and calculate_cosine_similarity(summary1, summary3) < 0.5 or calculate_cosine_similarity(summary1, summary2) == 1 and watermark in summary3:
                count += 1
        print(f"{language} asr is {count / 100}")
        # with open(f'{language}_val.txt', 'a') as file_watch:
        #     file_watch.write(f"{language} asr is {count / 100}\n")

