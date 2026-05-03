import streamlit as st
import cv2
import numpy as np

st.title("🍎 Fruit Ripeness Detection System")

# 📤 Upload Image
uploaded_file = st.file_uploader("Upload Fruit Image", type=["jpg", "png", "jpeg"])


# 🎯 Ripeness Function
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

    red_ratio = red_pixels / (total_pixels + 1)
    green_ratio = green_pixels / (total_pixels + 1)
    yellow_ratio = yellow_pixels / (total_pixels + 1)

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

    return label, score, red_ratio, green_ratio, yellow_ratio, color


# 🚀 PROCESS IMAGE
if uploaded_file is not None:

    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)

    st.image(img, caption="Uploaded Image", use_container_width=True)

    output = img.copy()
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Detect all fruits
    mask = cv2.inRange(hsv, (0,50,50), (180,255,255))

    kernel = np.ones((5,5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    results = []
    fruit_id = 1

    for cnt in contours:
        if cv2.contourArea(cnt) > 1000:

            x, y, w, h = cv2.boundingRect(cnt)

            label, score, r, g, yel, color = get_ripeness(hsv, cnt)

            # 🟩 Draw box
            cv2.rectangle(output, (x,y), (x+w,y+h), color, 2)

            # 🏷️ Label
            cv2.putText(output, f"{label} ({score:.2f})", (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            # 📊 Color ratio
            cv2.putText(output, f"R:{r:.2f} G:{g:.2f} Y:{yel:.2f}",
                        (x, y+h+20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2)

            # ✅ Store results
            results.append({
                "id": fruit_id,
                "label": label,
                "score": round(score, 2),
                "red": round(r, 2),
                "green": round(g, 2),
                "yellow": round(yel, 2)
            })

            fruit_id += 1

    # 🖥️ Show output image
    st.image(output, caption="Detected Fruits", use_container_width=True)

    # 📊 Show results in UI
    st.subheader("📊 Detection Results")

    for res in results:
        st.markdown(
            f"**Fruit {res['id']}** → {res['label']}  \n"
            f"Score: {res['score']}  \n"
            f"R:{res['red']} G:{res['green']} Y:{res['yellow']}"
        )

    # 🔢 Total count
    st.success(f"Total Fruits Detected: {len(results)}")