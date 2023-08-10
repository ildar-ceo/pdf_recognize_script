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
        
        if isinstance(size, BoxItem):
            
            self.x1 = size.x1
            self.x2 = size.x2
            self.y1 = size.y1
            self.y2 = size.y2
            
        else:
        
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
        
        self.w = self.x2 - self.x1
        self.h = self.y2 - self.y1
        
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
    
    def get_box_points(self):
        
        x1, y1, x2, y2 = self.x1, self.y1, self.x2, self.y2
        w = x2 - x1
        h = y2 - y1
        
        return [
            (x1, y1),
            (x1 + w // 2, y1),
            (x1 + w, y1),
            (x1 + w, y1 + h // 2),
            (x1 + w, y1 + h),
            (x1 + w // 2, y1 + h),
            (x1, y1 + h),
            (x1, y1 + h // 2),
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


def merge_lines(res):
    
    """
    Мерджит линии
    """
    
    return res


def get_paragraph_boxes(res, paragraph_threshold=25, line_threshold=10):
    
    """
    Функция получает параграфы, объединяя chars_boxes методом
    поиска в ширину, а также учитывая lines_boxes.
    """
    
    search = ParagraphSearcher()
    search.set_size(res["orig_image"].shape)
    search.set_mode("paragraph", paragraph_threshold, line_threshold)
    search.init_boxes(res["chars_boxes"])
    search.init_lines(res["lines_boxes"])
    search.get_all_paragraph()
    search.merge(5)
    
    res["paragraph_boxes"] = search.paragraphes
    res["paragraph_boxes_orig"] = search.paragraphes
    
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


class BoxHash:
    
    """
    Класс который кэширует прямоугольники
    для быстрого поиска ближайщих соседей
    """
    
    def __init__(self, items, kind, threshold):
        
        self.items = []
        self.items_hash_x = {}
        self.items_hash_y = {}
        self.points_hash = {}
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
        
        self.kind = kind
        self.items = list(items)
        
        for box_index in range(len(self.items)):
            
            box = self.items[box_index]
            
            if self.kind == "box":
                points = box.get_box_points()
                for point in points:
                    
                    x, y = point
                    self.add_point( x, y, box_index )
            
            if self.kind == "line":
                points = self.draw_line_bresenham(box, True)
                for point in points:
                    self.add_point(point[0], point[1], box_index)
    
    
    def add_point(self, x, y, box_index):
        
        """
        Добавить точку в кэш
        """
        
        x1 = x // self.threshold
        y1 = y // self.threshold
        
        if not(x1 in self.points_hash):
            self.points_hash[x1] = {}
        
        if not(y1 in self.points_hash[x1]):
            self.points_hash[x1][y1] = []
        
        self.points_hash[x1][y1].append( (x, y, box_index) )
    
    
    def get_nearest_items(self, box_index):
        
        """
        Функция возвращает ближающие прямоугольники
        рядом с box_index на расстоянии self.threshold
        """
        
        res = []
        nearest_search_points = []
        
        box_item = self.items[box_index]
        points = box_item.get_box_points()
        
        for point in points:
            
            x, y = point
            x1 = x // self.threshold 
            y1 = y // self.threshold
            
            for direction in self.directions:
                dx, dy = self.directions[direction]
                
                x2 = x1 + dx
                y2 = y1 + dy
                
                if not( (x2, y2) in nearest_search_points ):
                    nearest_search_points.append( (x, y, x2, y2) )
        
        for point in nearest_search_points:
            x, y, x2, y2 = point
            
            if x2 in self.points_hash:
                if y2 in self.points_hash[x2]:
                    
                    hash_points = self.points_hash[x2][y2]
                    for hash_point in hash_points:
                        
                        x3, y3, index = hash_point
                        box = self.items[index]
                        
                        d = math.sqrt((x3 - x) * (x3 - x) + (y3 - y) * (y3 - y))
                        
                        if d <= self.threshold:
                            if not(index in res) and box_index != index:
                                res.append( index )
        
        return res
    
    
    def draw_line_bresenham(self, line, is_bold=False):
        
        """
        Рисует линию алгоритмом Брезенхема
        """
        
        x1 = line.x1
        x2 = line.x2
        y1 = line.y1
        y2 = line.y2
        
        dx = abs(x2 - x1)
        dy = abs(y2 - y1)
        steep = dy > dx
        
        if steep:
            x1, y1 = y1, x1
            x2, y2 = y2, x2
        
        if x1 > x2:
            x1, x2 = x2, x1
            y1, y2 = y2, y1
        
        dx = abs(x2 - x1)
        dy = abs(y2 - y1)
        error = dx // 2
        y_step = self.threshold if y1 < y2 else -self.threshold
        
        y = y1
        points = []
        
        def add_point(point):
            if not(point in points):
                points.append(point)
        
        for x in range(x1, x2 + 1, self.threshold):
            
            if steep:
                add_point( (x, y) )
                if is_bold:
                    add_point( (x + self.threshold, y) )
                    add_point( (x - self.threshold, y) )
                    add_point( (x + self.threshold, y - self.threshold) )
                    add_point( (x - self.threshold, y - self.threshold) )
                    add_point( (x + self.threshold, y + self.threshold) )
                    add_point( (x - self.threshold, y + self.threshold) )
            else:
                add_point( (y, x) )
                if is_bold:
                    add_point( (y + self.threshold, x) )
                    add_point( (y - self.threshold, x) )
                    add_point( (y + self.threshold, x - self.threshold) )
                    add_point( (y - self.threshold, x - self.threshold) )
                    add_point( (y + self.threshold, x + self.threshold) )
                    add_point( (y - self.threshold, x + self.threshold) )
            
            error -= dy
            if error < 0:
                y += y_step
                error += dx
        
        return points
    
    
    def is_line_cross(self, line):

        """
        Проверяет пересекает ли линия line хотя бы одну линию из self.items
        """
        
        points = self.draw_line_bresenham(line)
        
        for point in points:
            x, y = point
            x = x // self.threshold
            y = y // self.threshold
            if x in self.points_hash:
                if y in self.points_hash[x]:
                    for hash_point in self.points_hash[x][y]:
                        _, _, index = hash_point
                        line2 = self.items[index]
                        if is_line_cross(line, line2):
                            return True
            
        return False


class ParagraphSearcher:
    
    """
    Класс, который объединяет боксы в параграфы методом поиска в ширину
    """
    
    def __init__(self):
        
        self.mode = None
        self.boxes = None
        self.boxes_matrix = None
        self.lines = None
        self.paragraphes = []
        self.current_line_y = None
        self.paragraph_threshold = 25
        self.line_threshold = 10
    
    
    def set_size(self, size):
        
        """
        Установить размер картинки
        """
        
        self.size = size
    
    
    def set_mode(self, mode, paragraph_threshold=25, line_threshold=10):
        
        """
        Устанавливает тип поиска:
        paragraph - искать параграфы
        line - искать линии
        """
        
        self.mode = mode
        self.paragraph_threshold = paragraph_threshold
        self.line_threshold = line_threshold
        
    
    def init_boxes(self, boxes):
        
        """
        Инициируем прямоугольники
        """
        
        if self.mode == "paragraph":
            boxes.sort(key = lambda box: box.w * box.h)
        
        elif self.mode == "line":
            boxes.sort(key = lambda box: (box.center_y, box.center_x))
        
        self.boxes = BoxHash(boxes, "box", self.paragraph_threshold)
        self.boxes_matrix = [1] * len(boxes)
    
    
    def init_lines(self, lines):
        
        """
        Инициируем линии
        """
        
        self.lines = BoxHash(lines, "line", self.paragraph_threshold)
    
    
    def get_paragraph(self, box_index):
        
        """
        Функция ищет параграф методом поиска в ширину, начиная с box_index
        """
        
        queue = collections.deque()
        queue.append( box_index )
        self.boxes_matrix[box_index] = 0
        
        box_item = self.boxes.items[box_index]
        paragraph = BoxItem( box_item )
        copy_paragraph = BoxItem( box_item )
        self.current_line_y = box_item.center_y
        
        while len(queue) > 0:
        
            box_index = queue.pop()
            box_item = self.boxes.items[box_index]
            box_item_center_point = box_item.get_box_center()
            
            # Расширить границы параграфа
            extend_paragraph(paragraph, box_item)
            
            neighbors_indexes = self.boxes.get_nearest_items(box_index)
            
            for index in neighbors_indexes:
                if self.boxes_matrix[index] == 1:
                    
                    neighbor_box_item = self.boxes.items[index]
                    neighbor_box_center_point = neighbor_box_item.get_box_center()
                    
                    line = BoxItem( (
                        box_item_center_point[0], box_item_center_point[1],
                        neighbor_box_center_point[0], neighbor_box_center_point[1]
                    ) )
                    
                    if self.mode == "line":
                        if abs(neighbor_box_item.center_y - self.current_line_y) \
                            > self.line_threshold:
                            break
                    
                    # Расширить границы параграфа
                    #copy_paragraph.resize( paragraph )
                    #extend_paragraph(copy_paragraph, neighbor_box_item)
                    
                    # Если новый параграф пересекает линию, то break
                    #if self.lines.is_line_cross(copy_paragraph):
                    if self.lines.is_line_cross(line):
                        break
                    
                    # Добавить в очередь
                    self.boxes_matrix[index] = 0
                    queue.append(index)
        
        return paragraph
    
    
    def get_next_index(self):
        
        box_index = -1
        
        try:
            box_index = self.boxes_matrix.index(1)
        except:
            pass
        
        return box_index
    
    
    def get_all_paragraph(self):
        
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
                
                if not self.lines.is_line_cross(line):
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


def extend_paragraph(paragraph, box):
    
    """
    Расширить параграф
    """
    
    x1 = paragraph.x1
    x2 = paragraph.x2
    y1 = paragraph.y1
    y2 = paragraph.y2
    
    if x1 > box.x1:
        x1 = box.x1
    if y1 > box.y1:
        y1 = box.y1
    if x2 < box.x2:
        x2 = box.x2
    if y2 < box.y2:
        y2 = box.y2
    
    paragraph.resize( (x1, y1, x2, y2) )
    

def merge_chars_boxes_to_lines(chars_boxes, line_threshold=5):
    
    """
    Объединение chars_boxes в линии
    """
    
    # Сортировка контуров по горизонтальной и вертикальной позиции
    chars_boxes = sorted(chars_boxes, key=lambda c: (c.center_y, c.center_x))
    
    paragraph = None
    paragraph_boxes = []
    
    # Итерация по контурам для сегментации
    for box in chars_boxes:
        
        if paragraph is not None and \
            box.center_y - prev_y > line_threshold:
            
            # Создать новую строку или абзац
            paragraph_boxes.append(paragraph)
            paragraph = None
        
        if paragraph is None:
            paragraph = BoxItem( box )
            
        else:
            # Расширить границы параграфа
            extend_paragraph(paragraph, box)
        
        prev_y = box.center_y
        prev_x = box.x2
    
    if paragraph is not None:
        paragraph_boxes.append(paragraph)
    
    return paragraph_boxes


def convert_big_paragraph_to_lines(res):
    
    """
    Конвертирует большие параграфы в линии
    """
    
    paragraph_boxes = []
    
    for box in res["paragraph_boxes"]:
        
        if box.w * box.h > 50000:
            
            chars_boxes = list(filter(
                lambda char_box: is_rectangle_cross(char_box, box),
                res["chars_boxes"]
            ))
            boxes = merge_chars_boxes_to_lines(chars_boxes)
            paragraph_boxes.extend( boxes )
        
        else:
            paragraph_boxes.append( box )
        
    res["paragraph_boxes"] = paragraph_boxes
    
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
    
    for j in range( max(index - distance, 0), min(index + distance + 1, len(boxes) - 1) ):
        box2 = boxes[j]
        if box2.category == 3:
            d = box_search_get_distance(box1, box2)
            if find_d is None or find_d > d:
                find_d = d
                find_box = box2
                find_index = j
    
    if find_d is None:
        return None
    
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
