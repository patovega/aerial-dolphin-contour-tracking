import cv2
import numpy as np
from collections import deque

def init_video(input_path, output_path):
   video = cv2.VideoCapture(input_path)
   width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
   height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
   fps = int(video.get(cv2.CAP_PROP_FPS))
   
   output = cv2.VideoWriter(output_path, 
                          cv2.VideoWriter_fourcc(*'mp4v'),
                          fps, 
                          (width, height))
   return video, output, width, height

def process_frame(frame):
   gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
   blur = cv2.GaussianBlur(gray, (5,5), 0)
   _, thresh = cv2.threshold(blur, 190, 255, cv2.THRESH_BINARY)
   return thresh

def filter_contours(contours):
   return [cnt for cnt in contours if cv2.contourArea(cnt) > 210 and 
           float(cv2.boundingRect(cnt)[2])/cv2.boundingRect(cnt)[3] <= 1.5]

def join_vertical_contours(contours, max_vertical=80, max_horizontal=15):
   unidos = []
   for cnt in contours:
       if not unidos:
           unidos.append(cnt)
           continue
           
       M1 = cv2.moments(cnt)
       if M1["m00"] == 0:
           continue
           
       cx1, cy1 = int(M1["m10"]/M1["m00"]), int(M1["m01"]/M1["m00"])
       unido = False
       
       for i, cnt_unido in enumerate(unidos):
           M2 = cv2.moments(cnt_unido)
           if M2["m00"] == 0:
               continue
               
           cx2, cy2 = int(M2["m10"]/M2["m00"]), int(M2["m01"]/M2["m00"])
           if abs(cy1-cy2) < max_vertical and abs(cx1-cx2) < max_horizontal:
               unidos[i] = np.vstack((cnt_unido, cnt))
               unido = True
               break
               
       if not unido:
           unidos.append(cnt)
           
   return unidos

def check_distance(cx, cy, cnt, max_dist=50):
   M = cv2.moments(cnt)
   if M["m00"] == 0:
       return False
   cx2, cy2 = int(M["m10"]/M["m00"]), int(M["m01"]/M["m00"])
   return np.sqrt((cx-cx2)**2 + (cy-cy2)**2) < max_dist

def check_persistence(cnt, buffer_contornos):
   M = cv2.moments(cnt)
   if M["m00"] == 0:
       return False
       
   cx, cy = int(M["m10"]/M["m00"]), int(M["m01"]/M["m00"])
   
   for frame_anterior in list(buffer_contornos)[:-1]:
       if not any(check_distance(cx, cy, cnt_ant) for cnt_ant in frame_anterior):
           return False
   return True

def get_stable_contours(contornos_unidos, buffer_contornos):
   if len(buffer_contornos) != buffer_contornos.maxlen:
       return []
       
   estables = []
   for cnt in contornos_unidos:
       if check_persistence(cnt, buffer_contornos):
           estables.append(cnt)
   return estables

def draw_labels(frame, num_delfines, max_delfines):
   overlay = frame.copy()
   cv2.rectangle(overlay, (30, 20), (250, 120), (0,0,0), -1)
   alpha = 0.7
   frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
   
   cv2.putText(frame, f"Actuales: {num_delfines}", (50, 50), 
              cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
   cv2.putText(frame, f"Maximo: {max_delfines}", (50, 100), 
              cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
   return frame

def main():
   video, output, _, _ = init_video('./video_delfines_2025.mp4', 
                                  'delfines_tracking_contornos.mp4')
   buffer_contornos = deque(maxlen=5)
   max_delfines = 0
   
   while True:
       ret, frame = video.read()
       if not ret:
           break
           
       thresh = process_frame(frame)
       contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, 
                                    cv2.CHAIN_APPROX_SIMPLE)
       
       filtrados = filter_contours(contours)
       unidos = join_vertical_contours(filtrados)
       buffer_contornos.append(unidos)
       estables = get_stable_contours(unidos, buffer_contornos)
       
       num_delfines = len(estables)
       max_delfines = max(max_delfines, num_delfines)
       
       result = frame.copy()
       cv2.drawContours(result, estables, -1, (0,255,0), 2)
       
       result = draw_labels(result, num_delfines, max_delfines)
       
       for cnt in estables:
           M = cv2.moments(cnt)
           if M["m00"] != 0:
               cx = int(M["m10"] / M["m00"])
               cy = int(M["m01"] / M["m00"])
               cv2.circle(result, (cx, cy), 5, (0,0,255), -1)
       
       output.write(result)

   print(f"Número máximo de delfines detectados: {max_delfines}")
   video.release()
   output.release()

if __name__ == "__main__":
   main()