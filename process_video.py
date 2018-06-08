import cv2
import numpy as np
from mrcnn.visualize_cv2 import model, display_instances, class_names

if __name__ == '__main__':
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Process video on trained Mask R-CNN model.')
    parser.add_argument('--video', required=True,
                        metavar="path or URL to video",
                        help='Video to apply')

    args = parser.parse_args()
    # Validate arguments
    if args.video:
        assert args.video, "Argument --video is required for training"

    print("Processing: ", args.video)

capture = cv2.VideoCapture(args.video)

size = (
    int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
    int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
)

codec = cv2.VideoWriter_fourcc(*'DIVX')

output = cv2.VideoWriter(args.video + '_masked.avi', codec, 60.0, size)

while(capture.isOpened()):
    
    ret, frame = capture.read()
    
    if ret:
        # Adding mask to frame
        results = model.detect([frame], verbose=1)
        r = results[0]

        frame = display_instances(
            frame, r['rois'], r['masks'], r['class_ids'], class_names, r['scores'],show_bbox=False
        )

        output.write(frame)
        cv2.imshow('frame', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    else:
        break

capture.release()
output.release()
cv2.destroyAllWindows()

