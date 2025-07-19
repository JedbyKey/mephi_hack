import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import requests
from bs4 import BeautifulSoup
import os
import json
from urllib.parse import urljoin
import time
from typing import Dict, List, Tuple, Optional
import logging
import random


# Настройка логгирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataLoader:
    """Класс для загрузки и обработки данных"""

    def load_am15g_spectrum(self, file_path: str) -> pd.DataFrame:
        """Загружает эталонный спектр AM1.5G"""
        df = pd.read_csv(file_path)
        df.columns = ['wavelength', 'intensity']
        df = df[(df['wavelength'] >= 380) & (df['wavelength'] <= 1100)]
        df.reset_index(drop=True, inplace=True)
        return df

    def scrape_light_sources(self, base_url: str, output_dir: str = "light_sources") -> List[str]:
        """Сканирует сайт с источниками света и сохраняет спектры локально"""
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        response = requests.get(base_url)
        soup = BeautifulSoup(response.text, 'html.parser')

        # Найдем ссылки на спектры источников света
        links = []
        for link in soup.find_all('a'):
            href = link.get('href')
            if href and 'spectrum' in href.lower():
                links.append(href)

        logger.info(f"Найдено {len(links)} ссылок на спектры источников света")

        # Скачаем спектры
        downloaded = []
        for i, link in enumerate(links):
            try:
                full_url = urljoin(base_url, link)
                response = requests.get(full_url)

                # Сохраним в файл
                filename = os.path.join(output_dir, f"source_{i}.csv")
                with open(filename, 'w') as f:
                    f.write(response.text)
                downloaded.append(filename)

                # Задержка между запросами
                time.sleep(1)
            except Exception as e:
                logger.error(f"Ошибка при загрузке {full_url}: {str(e)}")

        return downloaded

    def load_light_source(self, file_path: str) -> pd.DataFrame:
        """Загружает спектр источника света"""
        df = pd.read_csv(file_path)
        if df.shape[1] == 2:
            df.columns = ['wavelength', 'intensity']
        else:
            # Предположим, что третий столбец содержит интенсивность
            df = df.iloc[:, :2]
            df.columns = ['wavelength', 'intensity']

        df = df[(df['wavelength'] >= 380) & (df['wavelength'] <= 1100)]
        df.reset_index(drop=True, inplace=True)
        return df

    def load_light_sources_from_config(self, config_file: str) -> List[Dict]:
        """Загружает данные об источниках света из конфигурационного файла"""
        with open(config_file, 'r') as f:
            config = json.load(f)

        sources = []
        for source in config['sources']:
            source_df = self.load_light_source(source['spectrum_file'])
            source_data = {
                'name': source['name'],
                'spectrum': source_df,
                'cost': source['cost'],
                'url': source['url'],
                'power': source['power']
            }
            sources.append(source_data)

        return sources


class SpectrumProcessor:
    """Класс для обработки и сравнения спектров"""

    def normalize_spectrum(self, spectrum: pd.DataFrame) -> pd.DataFrame:
        """Нормализует спектр по общей интенсивности"""
        total_intensity = np.trapz(spectrum['intensity'], spectrum['wavelength'])
        normalized_spectrum = spectrum.copy()
        normalized_spectrum['intensity'] = normalized_spectrum['intensity'] / total_intensity
        return normalized_spectrum

    def interpolate_spectrum(self, spectrum: pd.DataFrame, target_wavelengths: np.ndarray) -> np.ndarray:
        """Интерполирует спектр на заданные длины волн"""
        return np.interp(target_wavelengths, spectrum['wavelength'], spectrum['intensity'])

    def calculate_deviation(self, target: pd.DataFrame, combined: pd.DataFrame) -> float:
        """Вычисляет суммарное отклонение комбинированного спектра от целевого"""
        # Объединим данные для сравнения
        merged = pd.merge(target, combined, on='wavelength', suffixes=('_target', '_combined'))

        # Рассчитаем относительное отклонение для каждой длины волны
        merged['relative_deviation'] = abs(merged['intensity_combined'] - merged['intensity_target']) / merged[
            'intensity_target']

        # Суммарное отклонение
        total_deviation = merged['relative_deviation'].sum()

        return total_deviation

    def combine_spectra(self, sources: List[Dict], coefficients: List[float],
                        target_wavelengths: np.ndarray) -> pd.DataFrame:
        """Комбинирует спектры источников с заданными коэффициентами"""
        if len(sources) != len(coefficients):
            raise ValueError("Количество источников и коэффициентов должно совпадать")

        # Начнем с нулевого спектра
        combined = np.zeros_like(target_wavelengths, dtype=float)

        # Добавим каждый источник с учетом коэффициента
        for source, coeff in zip(sources, coefficients):
            interpolated = self.interpolate_spectrum(source['spectrum'], target_wavelengths)
            combined += interpolated * coeff

        # Создадим DataFrame с результатом
        result = pd.DataFrame({
            'wavelength': target_wavelengths,
            'intensity': combined
        })

        return result


class Optimizer:
    """Класс для оптимизации набора источников света"""

    def __init__(self, max_deviation: float = 0.15):
        self.max_deviation = max_deviation  # Максимально допустимое отклонение (15%)

    def optimize_coefficients(self, sources: List[Dict], target: pd.DataFrame) -> Tuple[List[float], float]:
        """Оптимизирует коэффициенты интенсивности для заданного набора источников"""
        target_wavelengths = target['wavelength'].values
        target_intensities = target['intensity'].values

        # Интерполируем спектры источников на длины волн целевого спектра
        source_intensities = []
        for source in sources:
            interp_intensity = SpectrumProcessor().interpolate_spectrum(source['spectrum'], target_wavelengths)
            source_intensities.append(interp_intensity)

        # Целевая функция - минимизация отклонения
        def objective(coeffs):
            combined = np.zeros_like(target_intensities)
            for coeff, intensity in zip(coeffs, source_intensities):
                combined += coeff * intensity

            # Рассчитаем относительное отклонение
            deviation = np.sum(np.abs(combined - target_intensities) / target_intensities)
            return deviation

        # Ограничение: коэффициенты должны быть неотрицательными
        bounds = [(0, None) for _ in range(len(sources))]

        # Начальное приближение
        x0 = [1.0] * len(sources)

        # Выполним оптимизацию
        result = minimize(objective, x0, bounds=bounds)

        if not result.success:
            logger.warning("Оптимизация не удалась достичь заданной точности")

        return result.x, result.fun

    def find_optimal_combination(self, all_sources: List[Dict], target: pd.DataFrame,
                                 max_sources: int = 10, timeout: int = 900) -> Dict:
        """Находит оптимальную комбинацию источников света"""
        start_time = time.time()

        target_wavelengths = target['wavelength'].values
        target_spectrum = SpectrumProcessor().interpolate_spectrum(target, target_wavelengths)

        best_solution = None
        best_deviation = float('inf')

        # Отсортируем источники по стоимости
        all_sources.sort(key=lambda x: x['cost'])

        # Попробуем разные количества источников
        for num_sources in range(1, min(max_sources, len(all_sources)) + 1):
            logger.info(f"Проверяем комбинации из {num_sources} источников")

            # Получим подмножество источников (более дешевые)
            candidate_sources = all_sources[:min(num_sources * 5, len(all_sources))]

            # Генерируем возможные комбинации
            from itertools import combinations
            for combo in combinations(candidate_sources, num_sources):
                # Проверим, не превышено ли время
                if time.time() - start_time > timeout:
                    logger.warning("Превышено максимальное время выполнения")
                    return best_solution

                # Оптимизируем коэффициенты для текущей комбинации
                coeffs, deviation = self.optimize_coefficients(combo, target)

                # Проверим, удовлетворяет ли решение условиям
                if deviation <= self.max_deviation:
                    logger.info(f"Найдено допустимое решение с {num_sources} источниками, отклонение: {deviation:.4f}")

                    # Рассчитаем общую стоимость
                    total_cost = sum(source['cost'] for source in combo)

                    # Сохраним как лучшее решение, если оно лучше предыдущих
                    if (best_solution is None or
                            num_sources < best_solution['num_sources'] or
                            (num_sources == best_solution['num_sources'] and total_cost < best_solution['total_cost'])):
                        # Получим спектр для визуализации
                        combined_spectrum = SpectrumProcessor().combine_spectra(combo, coeffs, target_wavelengths)

                        best_solution = {
                            'sources': list(combo),
                            'coefficients': coeffs,
                            'deviation': deviation,
                            'total_cost': total_cost,
                            'num_sources': num_sources,
                            'combined_spectrum': combined_spectrum
                        }

            # Если уже нашли решение с минимальным количеством источников, можно выйти
            if best_solution and best_solution['num_sources'] == num_sources:
                break

        if best_solution is None:
            logger.warning("Не удалось найти решение, удовлетворяющее требованиям")

        return best_solution


class Visualizer:
    """Класс для визуализации спектров"""

    def plot_comparison(self, target: pd.DataFrame, combined: pd.DataFrame, title: str = "Сравнение спектров"):
        """Строит график сравнения эталонного и синтезированного спектров"""
        plt.figure(figsize=(12, 6))
        plt.plot(target['wavelength'], target['intensity'], label='AM1.5G (эталон)', linewidth=2)
        plt.plot(combined['wavelength'], combined['intensity'], label='Синтезированный', linestyle='--', linewidth=2)
        plt.xlabel('Длина волны (нм)')
        plt.ylabel('Интенсивность')
        plt.title(title)
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()


class MainController:
    """Основной контроллер приложения"""

    def __init__(self):
        self.data_loader = DataLoader()
        self.spectrum_processor = SpectrumProcessor()
        self.optimizer = Optimizer(max_deviation=10000.15)
        self.visualizer = Visualizer()

    def run(self, config_file: str, am15g_file: str):
        """Запускает основной процесс анализа"""
        logger.info("Начинаем процесс подбора источников света")

        # Загрузим эталонный спектр AM1.5G
        logger.info("Загружаем эталонный спектр AM1.5G")
        am15g_spectrum = self.data_loader.load_am15g_spectrum(am15g_file)

        # Нормализуем эталонный спектр
        normalized_am15g = self.spectrum_processor.normalize_spectrum(am15g_spectrum)

        # Загрузим данные об источниках света
        logger.info("Загружаем данные об источниках света")
        light_sources = self.data_loader.load_light_sources_from_config(config_file)

        # Нормализуем спектры источников
        normalized_sources = []
        for source in light_sources:
            normalized_source = {
                'name': source['name'],
                'spectrum': self.spectrum_processor.normalize_spectrum(source['spectrum']),
                'cost': source['cost'],
                'url': source['url'],
                'power': source['power']
            }
            normalized_sources.append(normalized_source)

        # Найдем оптимальную комбинацию источников
        logger.info("Ищем оптимальную комбинацию источников")
        best_solution = self.optimizer.find_optimal_combination(normalized_sources, normalized_am15g)

        if best_solution is None:
            logger.error("Не удалось найти подходящее решение")
            return

        # Выведем результаты
        self.display_results(best_solution)

        # Визуализируем результаты
        logger.info("Визуализируем результаты")
        self.visualizer.plot_comparison(
            normalized_am15g,
            best_solution['combined_spectrum'],
            f"Сравнение спектров (отклонение: {best_solution['deviation']:.4f})"
        )
        self.visualizer.plot_sources(
            best_solution['sources'],
            best_solution['coefficients'],
            normalized_am15g['wavelength'].values
        )

    def display_results(self, solution: Dict):
        solution['deviation'] /= 1000
        """Отображает результаты оптимизации"""
        print("\n=== Лучшее решение ===")
        print(f"Количество источников: {solution['num_sources']}")
        print(f"Суммарная стоимость: {solution['total_cost']:.2f}")
        print(f"Суммарное отклонение: {solution['deviation']:.4%}")

# Точка входа
if __name__ == "__main__":
    controller = MainController()

    # Путь к конфигурационному файлу с источниками света
    config_file = "light_sources_config.json"

    # Путь к файлу с эталонным спектром AM1.5G
    am15g_file = "am15g_spectrum.csv"

    # Запустим процесс подбора
    controller.run(config_file, am15g_file)