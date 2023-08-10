import io, sys, os, math, re
import collections
import subprocess
import matplotlib.pyplot as plt
import numpy as np
import pytesseract
import easyocr
import cv2


def show_image(image, name="", size=(30,20)):
    
    """
    Рисует изображение на экран
    """
    
    fig = plt.figure(figsize=size)
    plt.imshow(image, cmap="gray")
    if name != "":
        plt.title(name)
    plt.show()


def get_image_from_pdf(file_name, page_number):
    
    """
    Функция загружает изображения в Pillow по file_name, page_number
    """
    
    image = None
    
    # The pdftoppm command will convert each page to a separate image file
    command = [
       "pdftoppm",
        file_name,
        "-png",
        "-f", str(page_number),
        "-singlefile",
    ]
    
    try:
        output = subprocess.check_output(command)
        
        # Конвертация в формат Pillow
        #image_data = io.BytesIO(output)
        #image = Image.open(image_data)
        
        # Конвертируем картинку в image из opencv
        image_data = np.frombuffer(output, np.uint8)
        image = cv2.imdecode(image_data, cv2.IMREAD_COLOR)
        
    except subprocess.CalledProcessError as e:
        
        image = None
    
    return image


def descew_orig_image(res):
    
    import PIL
    from wand.image import Image
    
    image = res["orig_image"]
    
    with Image.from_array(image) as img_wand:
        
        #img_wand.deskew(0.4)
        img_wand.deskew(0.4 * img_wand.quantum_range)
        
        img_buffer = np.asarray(bytearray(img_wand.make_blob(format='png')), dtype='uint8')
        img_pil = PIL.Image.open( io.BytesIO(img_buffer) ).convert("RGB")
        image2 = np.array(img_pil)
    
    res["orig_image"] = image2
    
    return res
    

class BoxItem:
    
    def __init__(self, size):
        self.resize(size)
        self.text = None
        self.category = None
        self.category_rel = None
        self.category_predict = None
        self.recognize = None
        self.box_number = None
    
    def resize(self, size):
        
        if size[0] < size[2]:
            self.x1 = size[0]
            self.x2 = size[2]
        else:
            self.x2 = size[0]
            self.x1 = size[2]
        
        if size[1] < size[3]:
            self.y1 = size[1]
            self.y2 = size[3]
        else:
            self.y2 = size[1]
            self.y1 = size[3]
        
        self.w = size[2] - size[0]
        self.h = size[3] - size[1]
        
        self.center_x = (self.x1 + self.x2) // 2
        self.center_y = (self.y1 + self.y2) // 2
        
    def get_box_item(self):
        return (self.x1, self.y1, self.x2, self.y2)
    
    def get_box_center(self):
        return (self.center_x, self.center_y)
    
    def get_box_angles(self):
        return [
            (self.x1, self.y1),
            (self.x2, self.y1),
            (self.x2, self.y2),
            (self.x1, self.y2),
        ]
    
    def copy(self):
        item = BoxItem( self.get_box_item() )
        item.text = self.text
        item.category = self.category
        item.category_rel = self.category_rel
        item.category_predict = list(self.category_predict) \
            if self.category_predict is not None else None
        return item
        

def draw_boxes(image, boxes, color):
    
    """
    Рисует рамки на image
    """
    
    for box in boxes:
        cv2.rectangle(image, (box.x1, box.y1), (box.x2, box.y2), color, 2)


def get_words_boxes(image, mode=cv2.RETR_LIST, method=cv2.CHAIN_APPROX_SIMPLE):
    
    """
    Получает рамки на картинке
    """
    
    contours, _ = cv2.findContours(image, mode, method)
    boxes = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        boxes.append(BoxItem( (x,y,x+w,y+h) ))
        
    return boxes


def is_line_cross(line1, line2):
    
    """
    Проверяет пересекаются ли линии
    """
    
    a = np.array([[line1.x1, line1.y1], [line1.x2, line1.y2]])
    b = np.array([[line2.x1, line2.y1], [line2.x2, line2.y2]])
    
    v1 = np.cross(a[1] - a[0], b[0] - a[0])
    v2 = np.cross(a[1] - a[0], b[1] - a[0])
    
    if np.sign(v1) != np.sign(v2):
        v3 = np.cross(b[1] - b[0], a[0] - b[0])
        v4 = np.cross(b[1] - b[0], a[1] - b[0])
        
        if np.sign(v3) != np.sign(v4):
            return True
    
    return False


def is_rectangle_cross(A, B, threshold=0):
    
    """
    Функция определяет есть ли пересечение прямоугольников
    """
    
    Ax1, Ay1, Ax2, Ay2 = A.get_box_item()
    Bx1, By1, Bx2, By2 = B.get_box_item()
    
    Cx1 = max( min(Ax1, Ax2), min(Bx1, Bx2) )
    Cy1 = max( min(Ay1, Ay2), min(By1, By2) )
    
    Cx2 = min( max(Ax1, Ax2), max(Bx1, Bx2) )
    Cy2 = min( max(Ay1, Ay2), max(By1, By2) )
    
    Cx1 -= threshold
    Cy1 -= threshold
    
    Cx2 += threshold
    Cy2 += threshold
    
    if Cx1 >= Cx2:
        return False
    
    if Cy1 >= Cy2:
        return False
    
    return True


def is_rectangle_crossed(box, rectangles, threshold=0):
    
    """
    Проверяет пересекает ли box хотя бы один из
    прямоугольников rectangles
    """
    
    for index in range(len(rectangles)):
        res_box = rectangles[index]
        r = is_rectangle_cross(box, res_box, threshold)
        if r:
            return True
    
    return False


def is_line_rectangle_crossed(line, rectangles):
    
    """
    Проверяет пересекает ли линия хотя бы один прямоугольник
    """
    
    for index in range(len(rectangles)):
        res_box = rectangles[index]
        r = is_line_cross(line, res_box)
        if r:
            return True
    
    return False
    

def get_uncrossed_rectangles(boxes):
    
    """
    Возвращает box, которые не пеересекаются никаким другим box
    """
    
    boxes = sorted(boxes, \
        key=lambda box: box.w * box.h)
    
    result = []
    
    for box in boxes:
        is_crossed = is_rectangle_crossed(box, result)
        if not is_crossed:
            result.append(box)
    
    return result


def get_chars_boxes(res):
    
    """
    Функция возвращает регионы отдельных букв
    """
    
    # Конвертируем картинку в серый цвет
    threshval = 150
    gray_image = cv2.cvtColor(res["orig_image"], cv2.COLOR_BGR2GRAY)
    _, gray_image = cv2.threshold(gray_image, threshval, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    
    # Увеличиваем жирность
    kernel = np.ones((2, 2), np.uint8)
    dilated_image = cv2.dilate(gray_image, kernel, iterations=1)
    
    # Функция убирает маленькие прямоугольники,
    # которые скорее всего не являеются буквами
    def remove_small_box(box):
        w = box.w
        h = box.h
        if w * h < 20:
            return False
        return True
    
    # Функция убирает слишком большие прямоугольники,
    # которые скорее всего не являются буквами
    def remove_big_box(box):
        w = box.w
        h = box.h
        if w > 50 and h > 50 and w * h > 1000:
            return False
        return True
    
    # Функция убирает линии
    def remove_lines_box(box):
        w = box.w
        h = box.h
        if h < 10 and w > 30:
            return False
        if w < 10 and h > 30:
            return False
        return True
    
    # Получает boxes для каждой буквы
    res["chars_boxes_orig"] = get_words_boxes(
        cv2.bitwise_not(dilated_image),
        mode=cv2.RETR_LIST,
        method=cv2.CHAIN_APPROX_SIMPLE
        #method=cv2.CHAIN_APPROX_NONE
        #method=cv2.CHAIN_APPROX_TC89_L1
    )
    
    # chars_boxes_with_lines
    res["chars_boxes"] = res["chars_boxes_orig"]
    res["chars_boxes"] = list(filter(remove_big_box, res["chars_boxes"]))
    res["chars_boxes"] = list(filter(remove_lines_box, res["chars_boxes"]))
    res["chars_boxes"] = list(filter(remove_small_box, res["chars_boxes"]))
    
    res["gray_image"] = gray_image
    res["dilated_image"] = dilated_image
    
    return res


def get_lines_boxes(res):
    
    """
    Функция возвращает регионы линий.
    Сначала создает dilated_image, затем проходит свертками
    чтобы найти горизонтальные и вертикальные линии. Создает
    новое изображение из этих линий. А затем пробуе распознать
    эти линии с помощью cv2.HoughLinesP
    """
    
    """
    # Конвертируем картинку в серый цвет
    threshval = 150
    gray_image = cv2.cvtColor(res["orig_image"], cv2.COLOR_BGR2GRAY)
    _, gray_image = cv2.threshold(gray_image, threshval, 255, cv2.THRESH_BINARY_INV)
    
    # Увеличиваем жирность
    kernel = np.ones((2, 2), np.uint8)
    dilated_image = cv2.dilate(gray_image, kernel, iterations=1)
    """
    
    dilated_image = res["dilated_image"]
    
    # горизонтальные линии
    def show_horizontal_lines(image):
        hor = np.ones( (1, 30) )
        image = cv2.erode(image, hor, iterations=5)
        image = cv2.dilate(image, hor, iterations=5)
        return image
    
    # вертикальные линии
    def show_vertical_lines(image):
        ver = np.ones( (10, 1) )
        image = cv2.erode(image, ver, iterations=5)
        image = cv2.dilate(image, ver, iterations=5)
        return image
    
    horizontal_lines = show_horizontal_lines(dilated_image)
    vertical_lines = show_vertical_lines(dilated_image)
    
    # Сдвинуть горизонтальные линии на 5 пикселей
    rows, cols = horizontal_lines.shape
    shift_matrix = np.float32([[1, 0, -5], [0, 1, 0]])
    horizontal_lines = cv2.warpAffine(horizontal_lines, shift_matrix, (cols, rows))
    
    # Сдвинуть вертикальные линии на 5 пикселей
    rows, cols = vertical_lines.shape
    shift_matrix = np.float32([[1, 0, 0], [0, 1, -5]])
    vertical_lines = cv2.warpAffine(vertical_lines, shift_matrix, (cols, rows))
    
    # Объединить горизонтальные и вертикальные линии в одно изображение
    lines_clear_image = cv2.add(horizontal_lines, vertical_lines)
    
    # Обнаружение линий
    lines = cv2.HoughLinesP(
        lines_clear_image,
        rho=1, theta=np.pi / 500,
        threshold=20, minLineLength=20, maxLineGap=5)
    
    lines_boxes = []
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            lines_boxes.append( BoxItem(line[0]) )
    
    res["lines_boxes"] = lines_boxes
    res["lines_clear_image"] = lines_clear_image
    
    return res


def filter_chars_boxes(res):
    
    """
    Функция применяет фильтр к chars_boxes_with_lines
    и убирает прямоугольники, которые пересекаются с lines_boxes
    """
    
    lines_boxes = res["lines_boxes"]
    chars_boxes_with_lines = res["chars_boxes_with_lines"]
    
    def remove_crossed_boxes(box):
        if is_rectangle_crossed(box, lines_boxes):
            return False
        return True
    
    # Функция убирает слишком большие прямоугольники,
    # которые скорее всего не являеются буквами
    def remove_big_box(box):
        w = box.w
        h = box.h
        if w > 50 and h > 50 and w * h > 20000:
            return False
        return True
    
    res["chars_boxes"] = chars_boxes_with_lines
    #res["chars_boxes"] = list(filter(remove_crossed_boxes, res["chars_boxes"]))
    res["chars_boxes"] = list(filter(remove_big_box, res["chars_boxes"]))
    
    return res


def get_paragraph_boxes(res, threshold=25):
    
    """
    Функция получает параграфы, объединяя chars_boxes методом
    поиска в ширину, а также учитывая lines_boxes.
    """
    
    search = ParagraphSearcher(threshold)
    search.init_boxes(res["chars_boxes"])
    search.init_lines(res["lines_boxes"])
    search.get_all_paragraph()
    search.merge(5)
    
    res["paragraph_boxes"] = search.paragraphes
    
    return res


def sort_paragraph_boxes(res):
    
    """
    Сортирует параграфы по линиям
    """
    
    boxes = res["paragraph_boxes"]
    
    width = res["orig_image"].shape[1]
    arr = [ (index, box, box.get_box_center()) for index, box in enumerate(boxes) ]
    arr.sort( key=lambda item: item[2][1] * width + item[2][0] )
    
    res["paragraph_boxes"] = [ item[1] for item in arr ]
    
    return res
    

def recognize_paragraph_boxes(res, kind="easyocr", easyocr_reader=None):
    
    """
    Функция распознает текст в параграфах
    """
    
    image = res["orig_image"]
    for box in res["paragraph_boxes"]:
        cell = image[box.y1:box.y2, box.x1:box.x2]
        
        if kind == "easyocr":
            result = easyocr_reader.readtext(cell)
            box.recognize = result
            data = [text for (bbox, text, prob) in result]
            box.text = " ".join(data).lower()
        
        elif kind == "tesseract":
            text = pytesseract.image_to_string(cell, lang='kaz')
            box.text = text.strip().replace("\n", " ").lower()
    
    return res


class ParagraphSearcher:
    
    """
    Класс, который объединяет боксы в параграфы методом поиска в ширину
    """
    
    def __init__(self, paragraph_threshold, line_threshold):
        
        self.mode = None
        self.boxes = []
        self.boxes_hash = {}
        self.boxes_matrix = []
        self.lines = []
        self.lines_hash_x = {}
        self.lines_hash_y = {}
        self.line_threshold = line_threshold
        self.paragraphes = []
        self.paragraph_threshold = paragraph_threshold
        self.paragraph_threshold2 = paragraph_threshold * paragraph_threshold
        
        self.directions = {
            0: [0, 0],
            1: [-1, -1],
            2: [0, -1],
            3: [1, -1],
            4: [1, 0],
            5: [1, 1],
            6: [0, 1],
            7: [-1, 1],
            8: [-1, 0],
        }
    
    
    def init_boxes(self, boxes):
        
        """
        Инициируем прямоугольники
        """
        
        self.boxes = list(boxes)
        self.boxes_matrix = [1] * len(self.boxes)
        self.boxes.sort(key = lambda box: box.w * box.h)
        
        for box_index in range(len(self.boxes)):
            
            box = self.boxes[box_index]
            x1, y1, x2, y2 = box.get_box_item()
            w = x2 - x1
            h = y2 - y1
            
            self.addPoint( x1, y1, box_index )
            self.addPoint( x1 + w // 2, y1, box_index )
            self.addPoint( x1 + w, y1, box_index )
            self.addPoint( x1 + w, y1 + h // 2, box_index )
            self.addPoint( x1 + w, y1 + h, box_index )
            self.addPoint( x1 + w // 2, y1 + h, box_index )
            self.addPoint( x1, y1 + h, box_index )
            self.addPoint( x1, y1 + h // 2, box_index )
    
    
    def addPoint(self, x, y, box_index):
        
        x1 = x // self.paragraph_threshold
        y1 = y // self.paragraph_threshold
        
        if not(x1 in self.boxes_hash):
            self.boxes_hash[x1] = {}
        
        if not(y1 in self.boxes_hash[x1]):
            self.boxes_hash[x1][y1] = []
        
        if not((x, y) in self.boxes_hash[x1][y1]):
            self.boxes_hash[x1][y1].append( (x, y, box_index) )
    
    
    def init_lines(self, lines):
        
        """
        Инициируем линии
        """
        
        self.lines = list(lines)
        for line in self.lines:
            
            x1 = line.x1 // self.paragraph_threshold
            x2 = line.x2 // self.paragraph_threshold
            y1 = line.y1 // self.paragraph_threshold
            y2 = line.y2 // self.paragraph_threshold
            
            if not(x1 in self.lines_hash_x):
                self.lines_hash_x[x1] = []
            if not(x2 in self.lines_hash_x):
                self.lines_hash_x[x2] = []
            if not(y1 in self.lines_hash_y):
                self.lines_hash_y[y1] = []
            if not(y2 in self.lines_hash_y):
                self.lines_hash_y[y2] = []
            
            self.lines_hash_x[x1].append( line )
            self.lines_hash_x[x2].append( line )
            self.lines_hash_y[y1].append( line )
            self.lines_hash_y[y2].append( line )
    
    
    def get_lines_from_hash(self, line):
        
        """
        Возвращает список линий, находящихся в границах x1-x2 и y1-y2
        """
        
        x1 = line.x1 // self.paragraph_threshold
        x2 = line.x2 // self.paragraph_threshold
        y1 = line.y1 // self.paragraph_threshold
        y2 = line.y2 // self.paragraph_threshold
        
        res = []
        
        for x in range(x1, x2 + 1):
            if x in self.lines_hash_x:
                res.extend( self.lines_hash_x[x] )
        
        for y in range(y1, y2 + 1):
            if y in self.lines_hash_y:
                res.extend( self.lines_hash_y[y] )
        
        return res
    
    
    def is_lines_crossed(self, line):
        
        """
        Проверяет пересекает ли линия line
        ограничивающие линии self.lines
        """
        
        lines = self.get_lines_from_hash(line)
        is_line_crossed = is_line_rectangle_crossed(line, lines)
        
        return is_line_crossed
        
    
    def get_nearest_boxes(self, box_item):
        
        """
        Функция возвращает ближающие прямоугольники
        рядом с box_item на расстоянии self.paragraph_threshold
        """
        
        res = []
        x1, y1, x2, y2 = box_item.get_box_item()
        w = x2 - x1
        h = y2 - y1
        
        points = []
        points.append( (x1, y1) )
        points.append( (x1 + w // 2, y1) )
        points.append( (x1 + w, y1) )
        points.append( (x1 + w, y1 + h // 2) )
        points.append( (x1 + w, y1 + h) )
        points.append( (x1 + w // 2, y1 + h) )
        points.append( (x1, y1 + h) )
        points.append( (x1, y1 + h // 2) )
        
        nearest_search_points = []
        
        for point in points:
            
            x, y = point
            
            x1 = x // self.paragraph_threshold 
            y1 = y // self.paragraph_threshold
            
            for direction in self.directions:
                dx, dy = self.directions[direction]
                
                x2 = x1 + dx
                y2 = y1 + dy
                
                if not( (x2, y2) in nearest_search_points ):
                    nearest_search_points.append( (x2, y2) )
        
        for point in nearest_search_points:
            x2, y2 = point
            
            if x2 in self.boxes_hash:
                if y2 in self.boxes_hash[x2]:
                    hash_points = self.boxes_hash[x2][y2]
                    for hash_point in hash_points:
                        
                        x3, y3, box_index = hash_point
                        if (x3 - x) * (x3 - x) + (y3 - y) * (y3 - y) <= self.paragraph_threshold2:
                            if not(box_index in res):
                                res.append( box_index )
        
        return res
    
    
    def get_paragraph(self, box_index):
        
        """
        Функция ищет параграф методом поиска в ширину, начиная с box_index
        """
        
        queue = collections.deque()
        queue.append( box_index )
        self.boxes_matrix[box_index] = 0
        
        box_item = self.boxes[box_index]
        x1, y1, x2, y2 = box_item.get_box_item()
        paragraph = {
            "x1": x1,
            "y1": y1,
            "x2": x2,
            "y2": y2,
        }
        
        while len(queue) > 0:
        
            box_index = queue.pop()
            box_item = self.boxes[box_index]
            box_item_center_point = box_item.get_box_center()
            
            # Расширить границы параграфа
            if paragraph["x1"] > box_item.x1:
                paragraph["x1"] = box_item.x1
            if paragraph["y1"] > box_item.y1:
                paragraph["y1"] = box_item.y1
            if paragraph["x2"] < box_item.x2:
                paragraph["x2"] = box_item.x2
            if paragraph["y2"] < box_item.y2:
                paragraph["y2"] = box_item.y2
            
            neighbors_indexes = self.get_nearest_boxes(box_item)
            
            for index in neighbors_indexes:
                if self.boxes_matrix[index] == 1:
                    
                    neighbor_box_item = self.boxes[index]
                    neighbor_box_center_point = neighbor_box_item.get_box_center()
                    line = BoxItem( (
                        box_item_center_point[0], box_item_center_point[1],
                        neighbor_box_center_point[0], neighbor_box_center_point[1]
                    ) )
                    
                    if not self.is_lines_crossed(line):
                        
                        self.boxes_matrix[index] = 0
                        
                        # Добавить в очередь
                        queue.append(index)
        
        paragraph["w"] = paragraph["x2"] - paragraph["x1"]
        paragraph["h"] = paragraph["y2"] - paragraph["y1"]
        
        if paragraph["w"] <= 0 or paragraph["h"] <= 0:
            paragraph = None
    
        if paragraph is not None:
            paragraph = ( paragraph["x1"], paragraph["y1"], paragraph["x2"], paragraph["y2"] )
        
        return BoxItem(paragraph)
    
    
    def get_next_index(self):
        
        box_index = -1
        
        try:
            box_index = self.boxes_matrix.index(1)
        except:
            pass
        
        return box_index
    
    
    def get_all_paragraph(self):
        
        self.mode = "paragraph"
        self.paragraphes = []
        
        box_index = self.get_next_index()
        
        while box_index >= 0:
            
            paragraph = self.get_paragraph(box_index)
            
            if paragraph is not None:
                self.paragraphes.append(paragraph)
            
            box_index = self.get_next_index()
    
    
    def merge(self, distance=0):
        
        """
        Функция объединяет параграфы, которые расоложены рядом
        """
        
        boxes = sorted(self.paragraphes, reverse=True, \
            key=lambda box: box.w * box.h)
        
        merged_boxes = []
        for box in boxes:
            
            box_center_point = box.get_box_center()
            
            merged = False
            for index, merged_box in enumerate(merged_boxes):
                
                merged_box_center_point = merged_box.get_box_center()
                line = BoxItem( (box_center_point[0], box_center_point[1],
                    merged_box_center_point[0], merged_box_center_point[1]) )
                
                if not self.is_lines_crossed(line):
                    if is_rectangle_cross(box, merged_box, distance):
                        
                        x1 = min(box.x1, merged_box.x1)
                        y1 = min(box.y1, merged_box.y1)
                        x2 = max(box.x2, merged_box.x2)
                        y2 = max(box.y2, merged_box.y2)
                        
                        merged_boxes[index] = BoxItem( (x1, y1, x2, y2) )
                        merged = True
            
            if not merged:
                merged_boxes.append(box.copy())
        
        self.paragraphes = []
        for box in merged_boxes:
            if not(box in self.paragraphes):
                self.paragraphes.append(box)
    

# Ключевые слова для категории ФОТ
TEXT_CATEGORY_SUMMA = ['всех', 'годагодов', 'годов', 'годоф', 'гфзп',
    'гфот', 'еажқ', 'еңбекақ', 'жетқ', 'жфқ', 'заработн',
    'месячн', 'общ', 'оплат', 'плат', 'прем', 'работник',
    'размер', 'сақтандыр', 'сомас', 'составляет', 'страхов', 'сумм',
    'тенге', 'труд', 'төлеу', 'фзп', 'фонд', 'фот', 'қоры',
    'сомасы', 'теңге', 'тенге', 'обшая', 'обшее', 'жалпы', 'сыйлықақыс',
    'обцая', 'сактандыр', 'сыйлыкакыс'
]

# Ключевые слова для категории Кол-во сотрудников
TEXT_CATEGORY_WORKERS = ['единиц', 'застрахованн', 'застрахованн', 'к-во',
    'кол-во', 'количеств', 'общая', 'общее', 'персонал',
    'подлежащ', 'принят', 'раб-ков', 'работник', 'работниов',
    'расписан', 'сотрудник', 'страхован', 'страховател', 'человек',
    'численность', 'штат', 'саны', 'қызметкерлердің', 'адам', 'сақтанушы',
    'қызметкер'
]

# Ключевые слова для числа
TEXT_CATEGORY_NUMBER = [
    'одн', 'две', 'три', 'четыр', 'пят', 'шест', 'сем', 'восем',
    'девят', 'десят', 'двадцат', 'тридцат', 'сорок', 'девян',
    'сто', 'сот', 'двест', 'трист', 'четырест', 'тысяч', 'миллион',
    'екі', 'үш', 'төрт', 'бес', 'алты', 'жеті', 'сегіз', 'жиырм', 'алпыс', 'отыз',
    'жүз', 'мың', 'тиын', 'тенге', 'жетпіс', 'торт'
]

TEXT_NUMBERS = [
    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9'
]

def keywords_classification(text_arr, keywords):
    
    """
    Функция выводит релевантность класса текста
    на основе ключевых слов
    """
    
    text = " ".join(text_arr)
    words_count = len(text_arr)
    search_count = 0
    
    if words_count == 0:
        return 0
    
    for keyword in keywords:
        pos = text.find(keyword)
        if pos >= 0:
            search_count += 1
    
    return search_count / words_count


def word_is_number(word):
    
    """
    Определяет, является ли слово числом.
    Если там более 50% процентов цифр, то число.
    """
    
    rel = 0
    count = 0
    
    for ch in word:
        if ch in TEXT_NUMBERS:
            rel += 1
        count += 1
    
    if count == 0:
        return 0
    
    rel = rel / count
    
    if rel > 0.5:
        return True
    
    return False


def number_classification(text_arr):
    
    """
    Функция выводит релевантонсть, что это число
    """
    
    text = " ".join(text_arr)
    words_count = len(text_arr)
    search_count = 0
    
    if words_count == 0:
        return 0
    
    for keyword in TEXT_CATEGORY_NUMBER:
        pos = text.find(keyword)
        if pos >= 0:
            search_count += 1
    
    for word in text_arr:
        if word_is_number(word):
            search_count += 1
    
    rel = (search_count / words_count) * 1.2
    if rel > 1:
        rel = 1.0
    
    return rel


def get_text_words(text):
    words = re.split("\W", text)
    words = list(filter(lambda x: x != "", words))
    return words


def classify_the_text(res):
    
    """
    Определяет вероятность класса для региона
    """
    
    for box in res["paragraph_boxes"]:
        
        words = get_text_words(box.text)
        
        # Предсказываем класс
        predict = [0, 0, 0, 0]
        predict[1] = keywords_classification(words, TEXT_CATEGORY_SUMMA)
        predict[2] = keywords_classification(words, TEXT_CATEGORY_WORKERS)
        predict[3] = number_classification(words)
        
        box.category = 0
        box.category_rel = 0
        box.category_predict = predict
        
        # Максимальное предсказание
        index = np.argmax(predict)
        if predict[index] > 0.65:
            box.category = index
            box.category_rel = predict[index]
    
    return res


def get_all_boxes_text(res):
    
    data = {
        "Y, X": [],
        "Category": [],
        "Rel": [],
        "Text": [],
    }
    
    boxes = res["paragraph_boxes"]
    
    for box in boxes:
        box_center = box.get_box_center()
        data["Y, X"].append( (box_center[1], box_center[0]) )
        data["Category"].append(box.category)
        data["Rel"].append(box.category_rel)
        data["Text"].append(box.text)
    
    return data


def box_search_get_distance(box1, box2):
    
    """
    Сравнивает два региона и возвращает дистанцию между блоками
    """
    
    c1 = box1.get_box_angles()
    c2 = box2.get_box_angles()
    min_d = None
    
    for point1 in c1:
        for point2 in c2:
            d = math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)
            if min_d is None or min_d > d:
                min_d = d
    
    return min_d


def box_search_number(boxes, index, distance=10):
    
    box1 = boxes[index]
    find_box = None
    find_d = None
    find_index = None
    
    for j in range( min(index - distance, 0), max(index + distance + 1, len(boxes) - 1) ):
        box2 = boxes[j]
        if box2.category == 3:
            d = box_search_get_distance(box1, box2)
            if find_d is None or find_d > d:
                find_d = d
                find_box = box2
                find_index = j
    
    return {
        "box": find_box,
        "d": find_d,
        "index": find_index,
    }


def get_all_boxes_recognized_text(res):
    
    data = {
        "Index": [],
        "Category": [],
        "Rel": [],
        "Text": [],
        "Index2": [],
        "Number": [],
        "Distance": [],
    }
    
    boxes = res["paragraph_boxes"]
    
    for index, box in enumerate(boxes):
        if box.category in [1, 2]:
            
            box.box_number = box_search_number(boxes, index)
            
            data["Index"].append(index)
            data["Category"].append(box.category)
            data["Rel"].append(box.category_rel)
            data["Text"].append(box.text)
            
            if box.box_number is not None:
                data["Number"].append(box.box_number["box"].text)
                data["Distance"].append(box.box_number["d"])
                data["Index2"].append(box.box_number["index"])
            else:
                data["Number"].append(None)
                data["Distance"].append(None)
                data["Index2"].append(None)
    
    return data
