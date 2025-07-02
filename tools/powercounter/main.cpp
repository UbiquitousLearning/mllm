#include "Metrics.hpp"
#include <iostream>
#include <vector>
#include <cstring>
#include <unistd.h>
#include <sys/wait.h>
#include <thread>
#include <chrono>

using namespace std;
using namespace std::chrono;

int main(int argc, char **argv) {
    double sample_interval = 1.0;
    std::vector<std::string> cmd_args;

    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-t") == 0) {
            if (i + 1 < argc) {
                sample_interval = atof(argv[i + 1]);
                i++;
            } else {
                std::cerr << "Error: -t requires a value.\n";
                return 1;
            }
        } else {
            cmd_args.emplace_back(argv[i]);
        }
    }

    if (cmd_args.empty()) {
        std::cerr << "Error: No program specified.\n";
        return 1;
    }

    // Fork process
    pid_t pid = fork();
    if (pid == 0) {
        // Child process: execute the target program
        std::vector<char *> args;
        args.reserve(cmd_args.size());
        for (const auto &arg : cmd_args) { args.push_back(strdup(arg.c_str())); }
        args.push_back(nullptr);

        execvp(args[0], &args[0]);

        // Only reach here if execvp fails
        for (auto *arg : args) { free(arg); }
        perror("execvp failed");
        _exit(1);
    } else if (pid < 0) {
        perror("fork failed");
        return 1;
    }

    // Parent process: monitor power usage
    std::vector<double> samples;
    auto start_time = high_resolution_clock::now();

    while (true) {
        int status;
        pid_t result = waitpid(pid, &status, WNOHANG);
        if (result == pid) { break; }

        double power = powerNow();
        samples.push_back(power);

        std::this_thread::sleep_for(duration<double>(sample_interval));
    }

    // Ensure we wait for child to complete before exiting
    int status;
    waitpid(pid, &status, 0);

    auto end_time = high_resolution_clock::now();
    auto duration =
        std::chrono::duration_cast<std::chrono::duration<double>>(end_time - start_time).count();

    if (duration < 1e-9) {
        std::cerr << "The program executed too quickly to measure.\n";
        return 1;
    }

    double total_energy = 0.0;
    for (double p : samples) { total_energy += p * sample_interval; }

    double average_power = total_energy / duration;

    // Output results to console
    std::cout << "Total energy consumption: " << total_energy << " J\n";
    std::cout << "Average power: " << average_power << " W\n";

    return 0;
}