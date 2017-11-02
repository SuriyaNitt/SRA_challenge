#include "human_detection.h"
#include <fstream>

unordered_map parse_config(string fileName) {
    std::ifstream inputFile(fileName);
    unordered_map<string, double> inputs;

    std::string line;
    while (std::getline(inputFile, line)) {
        std::isstringstream iss(line);
        string key;
        double value;
        if (!(iss >> key >> value)) {break;}
        inputs[key] = value;
    }

    return inputs;
}

