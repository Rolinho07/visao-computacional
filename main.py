import cv2
import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog, messagebox


class FormDetector:
    def load_image(self, path_image):
        image = cv2.imread(path_image)
        if image is None:
            raise ValueError(f"Imagem não encontrada: {path_image}")
        return image

    def convert_to_grayscale(self, image):
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    def apply_blur(self, image):
        return cv2.GaussianBlur(image, (5, 5), 0)

    def edge_detection(self, image):
        return cv2.Canny(image, 50, 150)

    def find_lines(self, imagem_bordas):
        contornos, _ = cv2.findContours(
            imagem_bordas,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )
        return contornos

    def classify_forms(self, contornos, imagem_original):
        formas = {
            'triangulos': 0,
            'quadrados': 0,
            'circulos': 0,
            'pentagonos': 0,
            'outros': 0
        }

        for contorno in contornos:
            perimetro = cv2.arcLength(contorno, True)
            aproximacao = cv2.approxPolyDP(contorno, 0.02 * perimetro, True)

            if len(aproximacao) == 3:
                formas['triangulos'] += 1
                cv2.drawContours(imagem_original, [contorno], 0, (0, 255, 0), 2)

            elif len(aproximacao) == 4:
                formas['quadrados'] += 1
                cv2.drawContours(imagem_original, [contorno], 0, (0, 255, 0), 2)

            elif len(aproximacao) == 5:
                formas['pentagonos'] += 1
                cv2.drawContours(imagem_original, [contorno], 0, (0, 255, 0), 2)

            else:
                area = cv2.contourArea(contorno)
                perimetro_quadrado = perimetro ** 2
                circularity = (4 * np.pi * area) / perimetro_quadrado

                if circularity > 0.8:
                    formas['circulos'] += 1
                    cv2.drawContours(imagem_original, [contorno], 0, (0, 255, 0), 2)
                else:
                    formas['outros'] += 1
                    cv2.drawContours(imagem_original, [contorno], 0, (255, 255, 255), 2)

        return formas

    def view_results(self, imagem_original, formas):
        plt.figure(figsize=(10, 6))
        plt.imshow(cv2.cvtColor(imagem_original, cv2.COLOR_BGR2RGB))
        plt.title('Formas Geométricas Detectadas')
        plt.axis('off')

        legenda = f"""
Formas detectadas:
Triângulos: {formas['triangulos']}
Quadrados: {formas['quadrados']}
Círculos: {formas['circulos']}
Pentágonos: {formas['pentagonos']}
Outros: {formas['outros']}
"""
        plt.text(
            imagem_original.shape[1] * 0.05,
            imagem_original.shape[0] * 0.95,
            legenda,
            fontsize=10,
            verticalalignment='bottom',
            horizontalalignment='left',
            bbox=dict(facecolor='white', alpha=0.5)
        )

        plt.tight_layout()
        plt.show()


class Application(tk.Tk):
    def __init__(self):
        super().__init__()

        self.title("Detector de Formas")

        self.label = tk.Label(self, text="Selecione uma imagem para detectar formas geométricas:")
        self.label.pack(pady=20)

        self.button_load = tk.Button(self, text="Carregar Imagem", command=self.load_image)
        self.button_load.pack(pady=10)

    def load_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg;*.jpeg;*.png")])

        if file_path:
            detector = FormDetector()

            try:
                imagem = detector.load_image(file_path)
                imagem_cinza = detector.convert_to_grayscale(imagem)
                imagem_desfocada = detector.apply_blur(imagem_cinza)
                bordas = detector.edge_detection(imagem_desfocada)
                contornos = detector.find_lines(bordas)
                formas_detectadas = detector.classify_forms(contornos, imagem.copy())
                detector.view_results(imagem.copy(), formas_detectadas)

            except Exception as e:
                messagebox.showerror("Erro", f'Erro no processamento da imagem: {e}')


def main():
    app = Application()
    app.mainloop()


if __name__ == "__main__":
    main()