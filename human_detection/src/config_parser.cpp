#include "human_detection_private.h"
#include <fstream>

std::unordered_map<std::string, double> parse_config(std::string fileName) {
    std::ifstream inputFile(fileName);
    std::unordered_map<std::string, double> inputs;

    std::string line;
    while (std::getline(inputFile, line)) {
        std::istringstream iss(line);
        std::string key;
        double value;
        if (!(iss >> key >> value)) {break;}
        inputs[key] = value;
    }

    return inputs;
}

