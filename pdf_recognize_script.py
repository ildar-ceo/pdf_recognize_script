import io, sys, os, math, re
import collections
import subprocess
import matplotlib.pyplot as plt
import numpy as np
import pytesseract
import easyocr
import cv2
from PIL import Image, ImageDraw


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
        self.x1 = size[0]
        self.y1 = size[1]
        self.x2 = size[2]
        self.y2 = size[3]
        self.w = size[2] - size[0]
        self.h = size[3] - size[1]
        
    def get_box_item(self):
        return (self.x1, self.y1, self.x2, self.y2)
    
    def get_box_center(self):
        return (self.x1 + self.w // 2, self.y1 + self.h // 2)
    
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
        return None
    
    if Cy1 >= Cy2:
        return None
    
    x1 = min(Ax1, Bx1)
    y1 = min(Ay1, By1)
    x2 = max(Ax2, Bx2)
    y2 = max(Ay2, By2)
    
    return BoxItem( (x1, y1, x2, y2) )


def is_rectangle_crossed(box, rectangles, threshold=0):
    
    """
    Проверяет пересекает ли box хотя бы один из
    прямоугольников rectangles
    """
    
    is_crossed = False
    for index in range(len(rectangles)):
        res_box = rectangles[index]
        r = is_rectangle_cross(box, res_box, threshold)
        if r is not None:
            is_crossed = True
            break
    
    return is_crossed


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


def merge_rectangles(boxes, threshold=0):
    
    """
    Функция объединяет прямоугольники, которые расоложены рядом
    """
    
    res = []
    boxes = sorted(boxes, reverse=True, \
        key=lambda box: box.w * box.h)
    
    merged_boxes = []
    for box in boxes:
        
        merged = False
        for index, merged_box in enumerate(merged_boxes):
            res = is_rectangle_cross(box, merged_box, threshold)
            if res is not None:
                merged_boxes[index] = res
                merged = True
                #break
        
        if not merged:
            merged_boxes.append(box.copy())
    
    res = []
    for box in merged_boxes:
        if not(box in res):
            res.append(box)
    
    return res


def get_chars_boxes(orig_image):
    
    """
    Функция возвращает регионы отдельных букв
    """
    
    res = {}
    
    # Конвертируем картинку в серый цвет
    threshval = 150
    gray_image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2GRAY)
    _, gray_image = cv2.threshold(gray_image, threshval, 255, cv2.THRESH_BINARY_INV)
    
    # Увеличиваем жирность
    kernel = np.ones((2, 2), np.uint8)
    dilated_image = cv2.dilate(gray_image, kernel, iterations=1)
    
    # Функция убирает маленькие прямоугольники,
    # которые скорее всего не являеются буквами
    def remove_small_box(box):
        w = box.w
        h = box.h
        if w * h < 50:
            return False
        return True
    
    # Функция убирает большие прямоугольники,
    # которые скорее всего не являеются буквами
    def remove_big_box(box):
        w = box.w
        h = box.h
        if w > 30 and h > 30 and w * h > 2500:
            return False
        return True
    
    # Функция убирает линии
    def remove_lines_box(box):
        w = box.w
        h = box.h
        if h < 10:
            return False
        if w < 10 and h > 15:
            return False
        return True
    
    # Получает boxes для каждой буквы
    #res["chars_boxes_orig"] = get_words_boxes( dilated_image )
    res["chars_boxes_orig"] = get_words_boxes(
        cv2.bitwise_not(dilated_image),
        #mode=cv2.RETR_LIST,
        #method=cv2.CHAIN_APPROX_TC89_L1
    )
    res["chars_boxes_with_lines"] = list(filter(remove_small_box,  res["chars_boxes_orig"]))
    res["chars_boxes_without_lines"] = list(filter(remove_big_box,  res["chars_boxes_with_lines"]))
    res["chars_boxes_without_lines"] = list(filter(remove_lines_box,  res["chars_boxes_without_lines"]))
    
    res["orig_image"] = orig_image
    res["gray_image"] = gray_image
    res["dilated_image"] = dilated_image
    
    return res


def get_lines_boxes(res):
    
    """
    Функция возвращает регионы линий
    """
    
    dilated_image = res["dilated_image"]
    chars_boxes_without_lines = res["chars_boxes_without_lines"]
    
    # Создаем чистое изображение и закрашиваем буквы
    lines_clear_image = dilated_image.copy()
    for box in chars_boxes_without_lines:
        cv2.rectangle(lines_clear_image, (box.x1, box.y1), (box.x2, box.y2), 0, -1)
    
    # Обнаружение вертикальных линий
    vertical_lines = cv2.HoughLinesP(lines_clear_image, 1, np.pi / 90, \
        threshold=50, minLineLength=50, maxLineGap=10)
    
    # Обнаружение горизонтальных линий
    horizontal_lines = cv2.HoughLinesP(lines_clear_image, 1, np.pi / 90, \
        threshold=50, minLineLength=50, maxLineGap=10)
    
    lines_boxes = []
    
    if vertical_lines is not None:
        for line in vertical_lines:
            lines_boxes.append( BoxItem(line[0]) )
    
    if horizontal_lines is not None:
        for line in horizontal_lines:
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
    
    res["chars_boxes"] = list(filter(remove_crossed_boxes, chars_boxes_with_lines))
    res["chars_boxes"] = list(filter(remove_big_box, res["chars_boxes"]))
    
    return res


def get_paragraph_box(res):
    
    """
    Функция получает параграфы, объединяя chars_boxes методом
    поиска в ширину, а также учитывая lines_boxes.
    """
    
    threshold = 25
    search = ParagraphSearcher(threshold)
    search.init_boxes(res["chars_boxes"])
    search.init_lines(res["lines_boxes"])
    boxes = search.get_all_paragraph()
    boxes = merge_rectangles(boxes, 5)



def recognize_text_in_boxes(image, boxes, kind="tesseract", easyocr_reader=None):
    
    for box in boxes:
        cell = image[box.y1:box.y2, box.x1:box.x2]
        
        if kind == "easyocr":
            result = easyocr_reader.readtext(cell)
            box.recognize = result
            data = [text for (bbox, text, prob) in result]
            box.text = " ".join(data).lower()
        
        elif kind == "tesseract":
            text = pytesseract.image_to_string(cell, lang='kaz')
            box.text = text.strip().replace("\n", " ").lower()


class ParagraphSearcher:
    
    """
    Класс, который объединяет боксы в параграфы методом поиска в ширину
    """
    
    def __init__(self, threshold):
        
        self.boxes = []
        self.matrix = []
        self.lines = []
        self.hash = {}
        self.threshold = threshold
        self.threshold2 = threshold * threshold
        
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
        
        self.boxes = [ box.get_box_item() for box in boxes ]
        self.matrix = [1] * len(self.boxes)
        
        for box_index in range(len(self.boxes)):
            
            box = self.boxes[box_index]
            x1, y1, x2, y2 = box
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
    
    
    def init_lines(self, lines):
        
        """
        Инициируем линии
        """
        
        self.lines = [ lines.get_box_item() for line in lines ]
        
    
    def addPoint(self, x, y, box_index):
        
        x1 = x // self.threshold
        y1 = y // self.threshold
        
        if not(x1 in self.hash):
            self.hash[x1] = {}
        
        if not(y1 in self.hash[x1]):
            self.hash[x1][y1] = []
        
        if not((x, y) in self.hash[x1][y1]):
            self.hash[x1][y1].append( (x, y, box_index) )
    
    
    def get_nearest_boxes(self, box_item):
        
        res = []
        x1, y1, x2, y2 = box_item
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
            
            x1 = x // self.threshold 
            y1 = y // self.threshold
            
            for direction in self.directions:
                dx, dy = self.directions[direction]
                
                x2 = x1 + dx
                y2 = y1 + dy
                
                if not( (x2, y2) in nearest_search_points ):
                    nearest_search_points.append( (x2, y2) )
        
        for point in nearest_search_points:
            x2, y2 = point
            
            if x2 in self.hash:
                if y2 in self.hash[x2]:
                    hash_points = self.hash[x2][y2]
                    for hash_point in hash_points:
                        
                        x3, y3, box_index = hash_point
                        if (x3 - x) * (x3 - x) + (y3 - y) * (y3 - y) <= self.threshold2:
                            if not(box_index in res):
                                res.append( box_index )
        
        return res
    
    
    def get_paragraph(self, box_index):
        
        queue = collections.deque()
        queue.append( box_index )
        self.matrix[box_index] = 0
        
        box_item = self.boxes[box_index]
        x1, y1, x2, y2 = box_item
        paragraph = {
            "x": x1,
            "y": y1,
            "right": x2,
            "bottom": y2,
        }
        
        while len(queue) > 0:
        
            box_index = queue.pop()
            box_item = self.boxes[box_index]
           
            x1, y1, x2, y2 = box_item
            
            if paragraph["x"] > x1:
                paragraph["x"] = x1
            if paragraph["y"] > y1:
                paragraph["y"] = y1
            if paragraph["right"] < x2:
                paragraph["right"] = x2
            if paragraph["bottom"] < y2:
                paragraph["bottom"] = y2
            
            neighbors_indexes = self.get_nearest_boxes(box_item)
            
            for index in neighbors_indexes:
                if self.matrix[index] == 1:
                    self.matrix[index] = 0
                    queue.append(index)
        
        paragraph["w"] = paragraph["right"] - paragraph["x"]
        paragraph["h"] = paragraph["bottom"] - paragraph["y"]
        
        if paragraph["w"] <= 0 or paragraph["h"] <= 0:
            paragraph = None
    
        if paragraph is not None:
            paragraph = ( paragraph["x"], paragraph["y"], paragraph["right"], paragraph["bottom"] )
        
        return BoxItem(paragraph)
    
    
    def get_next_index(self):
        
        box_index = -1
        
        try:
            box_index = self.matrix.index(1)
        except:
            pass
        
        return box_index
    
    
    def get_all_paragraph(self):
        
        res = []
        
        box_index = self.get_next_index()
        
        while box_index >= 0:
            
            paragraph = self.get_paragraph(box_index)
            
            if paragraph is not None:
                res.append(paragraph)
            
            box_index = self.get_next_index()
            
        return res


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


def box_classification(box):
    
    """
    Определяет вероятность класса для региона
    """
    
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


def box_search_get_distance(box1, box2, kind):
    
    """
    Сравнивает два региона и возвращает дистанцию между блоками
    Или -1, если два региона не относятся к kind
    """
    
    if kind == "right":
        
        if box2.x1 < box1.x2 - 10:
            return -1
        
        if box2.y2 < box1.y1:
            return -1
        
        if box2.y1 > box1.y2:
            return -1
    
    if kind == "bottom":
        
        if box2.y1 < box1.y2 - 10:
            return -1
        
        if box2.x2 < box1.x1:
            return -1
        
        if box2.x1 > box1.x2:
            return -1
    
    c1 = box1.get_box_center()
    c2 = box2.get_box_center()
    
    d = math.sqrt((c1[0] - c2[0])**2 + (c1[1] - c2[1])**2)
    return d


def box_search_number_kind(boxes, index, kind):
    
    """
    Возвращает index ближайшего блока с числом справа или снизу.
    """
    
    box1 = boxes[index]
    find_box = None
    find_d = None
    
    for box2_index in range(len(boxes)):
        box2 = boxes[box2_index]
        if index != box2_index:
            d = box_search_get_distance(box1, box2, kind)
            if d >= 0 and box2.category == 3:
                if find_d is None or find_d > d:
                    find_d = d
                    find_box = box2
    
    if find_box and find_box.category != 3:
        find_box = None
        find_d = None
    
    return find_box, find_d
    

def box_search_number(boxes, index):
    
    """
    Возвращает index ближайшего блока с числом
    """
    
    box_right, box_right_d = box_search_number_kind(boxes, index, "right")
    box_bottom, box_bottom_d = box_search_number_kind(boxes, index, "bottom")
    
    if box_right_d is not None and box_bottom_d is not None:
        if box_right_d < box_bottom_d:
            return box_right, box_right_d, "right"
        else:
            return box_bottom, box_bottom_d, "bottom"
    
    if box_right_d is not None:
        return box_right, box_right_d, "right"
    
    if box_bottom_d is not None:
        return box_bottom, box_bottom_d, "bottom"
    
    return None


def get_all_boxes_text(boxes):
    
    data = {
        "Index": [],
        "Category": [],
        "Rel": [],
        "Text": [],
    }

    for index in range(len(boxes)):
        box = boxes[index]
        data["Index"].append(index)
        data["Category"].append(box.category)
        data["Rel"].append(box.category_rel)
        data["Text"].append(box.text)
    
    return data


def get_all_boxes_recognized_text(boxes):
    
    data = {
        "Index": [],
        "Category": [],
        "Rel": [],
        "Text": [],
        "Number": [],
        "Distance": [],
        "Kind": [],
    }

    for index in range(len(boxes)):
        box = boxes[index]
        if box.category in [1, 2]:
            
            box.box_number = box_search_number(boxes, index)
            
            data["Index"].append(index)
            data["Category"].append(box.category)
            data["Rel"].append(box.category_rel)
            data["Text"].append(box.text)
            
            if box.box_number is not None:
                data["Number"].append(box.box_number[0].text)
                data["Distance"].append(box.box_number[1])
                data["Kind"].append(box.box_number[2])
            else:
                data["Number"].append(None)
                data["Distance"].append(None)
                data["Kind"].append(None)
    
    return data
