#include <iostream>
#include <fstream>
#include <string>
#include <unordered_map>
#include <curl/curl.h>
#include <json/json.h>

// Alpaca API credentials
const std::string API_KEY = "YOUR_ALPACA_API_KEY";
const std::string SECRET_KEY = "YOUR_ALPACA_SECRET_KEY";
const std::string BASE_URL = "https://paper-api.alpaca.markets";

// Function to read sentiment scores from file
std::unordered_map<std::string, double> readSentimentScores(const std::string &filename) {
    std::unordered_map<std::string, double> sentimentScores;
    std::ifstream file(filename);
    std::string line;
    while (std::getline(file, line)) {
        size_t commaPos = line.find(',');
        std::string symbol = line.substr(0, commaPos);
        double score = std::stod(line.substr(commaPos + 1));
        sentimentScores[symbol] = score;
    }
    return sentimentScores;
}

// Function to determine trade action based on sentiment score
std::string determineAction(double sentimentScore, double threshold = 0.1) {
    if (sentimentScore > threshold) {
        return "buy";
    } else if (sentimentScore < -threshold) {
        return "sell";
    } else {
        return "hold";
    }
}

// Function to determine trade quantity based on sentiment score
int determineQuantity(double sentimentScore, int baseQuantity = 10) {
    return static_cast<int>(baseQuantity * std::abs(sentimentScore) * 10);
}

// Function to execute trade
void executeTrade(const std::string &action, const std::string &symbol, int quantity) {
    CURL *curl;
    CURLcode res;

    curl_global_init(CURL_GLOBAL_DEFAULT);
    curl = curl_easy_init();

    if (curl) {
        std::string url = BASE_URL + "/v2/orders";
        std::string data = "{\"symbol\":\"" + symbol + "\",\"qty\":" + std::to_string(quantity) +
                           ",\"side\":\"" + action + "\",\"type\":\"market\",\"time_in_force\":\"gtc\"}";

        struct curl_slist *headers = NULL;
        headers = curl_slist_append(headers, ("APCA-API-KEY-ID: " + API_KEY).c_str());
        headers = curl_slist_append(headers, ("APCA-API-SECRET-KEY: " + SECRET_KEY).c_str());
        headers = curl_slist_append(headers, "Content-Type: application/json");

        curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
        curl_easy_setopt(curl, CURLOPT_POSTFIELDS, data.c_str());
        curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);

        res = curl_easy_perform(curl);

        if (res != CURLE_OK) {
            std::cerr << "curl_easy_perform() failed: " << curl_easy_strerror(res) << std::endl;
        }

        curl_slist_free_all(headers);
        curl_easy_cleanup(curl);
    }

    curl_global_cleanup();
}

int main() {
    const std::string sentimentFile = "sentiment_scores.txt";
    const int interval = 60; // in seconds

    while (true) {
        auto sentimentScores = readSentimentScores(sentimentFile);

        for (const auto &entry : sentimentScores) {
            const std::string &symbol = entry.first;
            double sentimentScore = entry.second;

            std::string action = determineAction(sentimentScore);
            if (action != "hold") {
                int quantity = determineQuantity(sentimentScore);
                executeTrade(action, symbol, quantity);
                std::cout << "Executed " << action << " trade for " << symbol << " with quantity " << quantity << std::endl;
            } else {
                std::cout << "No significant sentiment change detected for " << symbol << ", holding position." << std::endl;
            }
        }

        std::this_thread::sleep_for(std::chrono::seconds(interval));
    }

    return 0;
}
