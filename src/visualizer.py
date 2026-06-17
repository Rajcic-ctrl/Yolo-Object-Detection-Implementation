import cv2


def draw_counts_overlay(frame, object_counts: dict[str, int]):
    if not object_counts:
        return frame

    lines = ["Detected objects"]

    for class_name, count in sorted(object_counts.items()):
        lines.append(f"{class_name}: {count}")

    x = 15
    y = 15
    padding = 10
    line_height = 28
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7
    thickness = 2

    text_widths = [
        cv2.getTextSize(line, font, font_scale, thickness)[0][0]
        for line in lines
    ]

    box_width = max(text_widths) + padding * 2
    box_height = line_height * len(lines) + padding

    overlay = frame.copy()

    cv2.rectangle(
        overlay,
        (x, y),
        (x + box_width, y + box_height),
        (0, 0, 0),
        -1
    )

    cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)

    for index, line in enumerate(lines):
        text_y = y + padding + 20 + index * line_height

        cv2.putText(
            frame,
            line,
            (x + padding, text_y),
            font,
            font_scale,
            (255, 255, 255),
            thickness,
            cv2.LINE_AA
        )

    return frame