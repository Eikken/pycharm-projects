from PyPDF2 import PdfFileWriter, PdfFileReader
import os

path = r"C:\Users\Celeste\Desktop\申报论文材料"
output = PdfFileWriter()


def split_pdf(result):
    """从filename中提取[start,end)之间的页码内容保存为result"""
    # 打开原始 pdf 文件
    with open(result, "wb") as fp:
        pdf = PdfFileWriter()
        for p in range(30):
            pdfPath = path + '\\' + str(p + 1) + '.pdf'
            print(pdfPath)
            pdf_src = PdfFileReader(pdfPath)
            # 创建空白pdf文件
            # 提取页面内容，写入空白文件
            pdf.addPage(pdf_src.getPage(0))
            # 写入结果pdf
            pdf.write(fp)


split_pdf("30.pdf")

print('finish')