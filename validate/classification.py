from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
from tqdm import tqdm


def create_pdf(results, pdf_name):
    with PdfPages(pdf_name) as pdf:
        for el in results:
            image = el["image"]
            score = el["score"]
            url = el["url"]
            plt.figure(figsize=(20, 20))
            plt.title(f"{score}\n {url}")
            plt.imshow(image)
            pdf.savefig()
            plt.clf()
            plt.close()
