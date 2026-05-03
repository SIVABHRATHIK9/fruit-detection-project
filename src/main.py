import cv2
import os
import numpy as np

# 📂 Use a folder with MULTIPLE fruit images (IMPORTANT)
folder = folder = r"C:\Users\sivab\OneDrive\Desktop\Fruit_Project\dataset\fruits-360_3-body-problem\fruits-360-3-body-problem\Training\Apple\Apple Red 2"

print("Path exists:", os.path.exists(folder))

# 🎯 Improved ripeness function
def get_ripeness(hsv, contour):
    mask = np.zeros(hsv.shape[:2], dtype="uint8")
    cv2.drawContours(mask, [contour], -1, 255, -1)

    total_pixels = cv2.countNonZero(mask)

    # 🔴 Red
    red_mask = cv2.inRange(hsv, (0,120,70), (10,255,255)) + \
               cv2.inRange(hsv, (170,120,70), (180,255,255))

    # 🟢 Green
    green_mask = cv2.inRange(hsv, (35,50,50), (85,255,255))

    # 🟡 Yellow
    yellow_mask = cv2.inRange(hsv, (20,100,100), (35,255,255))

    red_pixels = cv2.countNonZero(cv2.bitwise_and(red_mask, red_mask, mask=mask))
    green_pixels = cv2.countNonZero(cv2.bitwise_and(green_mask, green_mask, mask=mask))
    yellow_pixels = cv2.countNonZero(cv2.bitwise_and(yellow_mask, yellow_mask, mask=mask))

    # 📊 Ratios
    red_ratio = red_pixels / (total_pixels + 1)
    green_ratio = green_pixels / (total_pixels + 1)
    yellow_ratio = yellow_pixels / (total_pixels + 1)

    # 🎯 Improved scoring
    score = (red_ratio * 10) + (yellow_ratio * 5)

    if red_ratio > 0.6:
        label = "Ripe"
        color = (0,255,0)
    elif green_ratio > 0.5:
        label = "Unripe"
        color = (0,0,255)
    else:
        label = "Semi-Ripe"
        color = (0,255,255)

    return label, round(score,2), red_ratio, green_ratio, yellow_ratio, color


# 📂 Load images
files = os.listdir(folder)

for file in files:
    path = os.path.join(folder, file)

    img = cv2.imread(path)
    if img is None:
        continue

    output = img.copy()

    # 🎨 Convert to HSV
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # 🔗 Combined mask (detect all fruits)
    mask = cv2.inRange(hsv, (0,50,50), (180,255,255))

    # 🧼 Clean mask
    kernel = np.ones((5,5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # 🔍 Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    fruit_id = 1

    for cnt in contours:
        area = cv2.contourArea(cnt)

        if area > 1000:   # bigger threshold for real images
            x, y, w, h = cv2.boundingRect(cnt)

            label, score, r, g, yel, color = get_ripeness(hsv, cnt)

            # 🟩 Draw box
            cv2.rectangle(output, (x,y), (x+w,y+h), color, 2)

            # 🏷️ Label
            cv2.putText(output, f"{label} ({score})", (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            # 📊 Colour percentage
            percent_text = f"R:{r:.2f} G:{g:.2f} Y:{yel:.2f}"
            cv2.putText(output, percent_text, (x, y+h+20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2)

            print(f"Fruit {fruit_id} → {label} | Score: {score} | R:{r:.2f} G:{g:.2f} Y:{yel:.2f}")

            fruit_id += 1

    total = fruit_id - 1
    print("Total Fruits:", total)

    cv2.putText(output, f"Total: {total}", (20,40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

    # 🖥️ Show
    cv2.imshow("Output", output)

    if cv2.waitKey(0) == 27:
        break
cv2.destroyAllWindows()