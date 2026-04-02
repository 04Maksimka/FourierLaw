#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <iomanip>
#include <algorithm>

double M_PI = 3.14159;

// Структура для хранения конфигурации знаков
struct SignConfig {
    int s0, s1, s2, s3;
};

// функция для приведения индекса к диапазону [-l, l]
int normalize_index(int k, int l) {
    int L = 2 * l + 1;
    int k_mod = k % L;
    if (k_mod < 0) k_mod += L;
    if (k_mod > l) k_mod -= L;
    return k_mod;
}

int main() {
    std::ofstream outfile("resonance_analysis.txt");
    
    if (!outfile.is_open()) {
        std::cerr << "Error: Could not open output file." << std::endl;
        return 1;
    }

    // Заголовок таблицы
    outfile << std::setw(6) << "L" 
            << std::setw(18) << "Total_Resonances" 
            << std::setw(18) << "None_Trivial"
            << std::setw(18) << "Trivial"
            << std::setw(12) << "Density" << std::endl;
            
    std::cout << "Starting calculation..." << std::endl;

    // Предопределяем 6 возможных конфигураций знаков (сумма сигнатур = 0)
    // Порядок знаков: sigma, sigma1, sigma2, sigma3
    // Условие резонанса: sigma*w + sigma1*w1 + sigma2*w2 + sigma3*w3 = 0
    std::vector<SignConfig> signs = {
        {+1, +1, -1, -1},
        {+1, -1, +1, -1},
        {+1, -1, -1, +1},
        {-1, +1, +1, -1},
        {-1, +1, -1, +1},
        {-1, -1, +1, +1}
    };

    // Цикл по размерам решётки L
    for (int l = 1; l <= 1001; ++l) {
        int L = 2 * l + 1;
        
        // Предвычисление частот
        std::vector<double> omega(L);
        for (int k = -l; k <= l; ++k) {
            double arg = M_PI * k / L;
            omega[k + l] = std::sqrt(1.0 + 4.0 * std::pow(std::sin(arg), 2));
        }

        long long total_resonances = 0;
        long long trivial_resonances = 0;
        long long total_combinations = 0; // Для справки (L^3 * 6)

        // Основные циклы
        for (int k = -l; k <= l; ++k) {
            for (int k1 = -l; k1 <= l; ++k1) {
                for (int k2 = -l; k2 <= l; ++k2) {
                    
                    // k + k1 + k2 + k3 = 0
                    int k3_raw = -(k + k1 + k2);
                    int k3 = normalize_index(k3_raw, l);

                    double w  = omega[k + l];
                    double w1 = omega[k1 + l];
                    double w2 = omega[k2 + l];
                    double w3 = omega[k3 + l];

                    // Проверяем 6 конфигураций знаков
                    for (const auto& cfg : signs) {
                        total_combinations++;

                        double sum = cfg.s0 * w + cfg.s1 * w1 + cfg.s2 * w2 + cfg.s3 * w3;
                        
                        if (std::abs(sum) < 1e-9) {
                            total_resonances++;

                            // Проверка на тривиальность.
                            // Тривиальный резонанс: множество индексов с '+' совпадает с множеством индексов с '-'
                            // Это означает, что члены с '+' и '-' попарно сокращаются.
                            
                            std::vector<int> pos_indices;
                            std::vector<int> neg_indices;

                            // Сортируем индексы по знакам
                            if (cfg.s0 > 0) pos_indices.push_back(k);  else neg_indices.push_back(k);
                            if (cfg.s1 > 0) pos_indices.push_back(k1); else neg_indices.push_back(k1);
                            if (cfg.s2 > 0) pos_indices.push_back(k2); else neg_indices.push_back(k2);
                            if (cfg.s3 > 0) pos_indices.push_back(k3); else neg_indices.push_back(k3);

                            // Сортируем векторы для корректного сравнения множеств
                            std::sort(pos_indices.begin(), pos_indices.end());
                            std::sort(neg_indices.begin(), neg_indices.end());

                            if (pos_indices == neg_indices) {
                                trivial_resonances++;
                            }
                        }
                    }
                }
            }
        }

        long long non_trivial = total_resonances - trivial_resonances;
        double density = static_cast<double>(total_resonances) / total_combinations;

        outfile << std::setw(6) << L 
                << std::setw(18) << total_resonances 
                << std::setw(18) << trivial_resonances
                << std::setw(18) << non_trivial
                << std::setw(12) << std::fixed << std::setprecision(8) << density 
                << std::endl;

        std::cout << "Processed L = " << L << ". Total: " << total_resonances 
                  << ", Trivial: " << trivial_resonances << ", Non-trivial: " << non_trivial << std::endl;
    }

    outfile.close();
    std::cout << "Done. Results in 'resonance_analysis.txt'." << std::endl;

    return 0;
}