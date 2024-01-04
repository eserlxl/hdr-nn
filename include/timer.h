#ifndef TIMER_H
#define TIMER_H

#include <iostream>
#include <chrono>

struct Timer {
    explicit Timer(const char *label) {
        m_start = std::chrono::high_resolution_clock::now();
        m_label = label;
    }

    ~Timer() {
        std::chrono::duration<float> seconds = std::chrono::high_resolution_clock::now() - m_start;
        std::cout << m_label << seconds.count() << " seconds" << std::endl;
    }

    std::chrono::high_resolution_clock::time_point m_start;
    const char *m_label;
};

#endif //TIMER_H