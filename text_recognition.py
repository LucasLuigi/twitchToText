from imutils.object_detection import non_max_suppression
import numpy as np
import pytesseract
import argparse
import cv2


def load_image(image_path: str):
    image_obj = cv2.imread(args.image)
    return image_obj


def resize_image(image_obj: cv2.Mat, original_width: int, original_height: int, configured_new_width: int, configured_new_height: int):
    resized_image_obj = cv2.resize(
        image_obj, (configured_new_width, configured_new_height))
    return resized_image_obj


def decode_predictions(scores, geometry):
    # grab the number of rows and columns from the scores volume, then
    # initialize our set of bounding box rectangles and corresponding
    # confidence scores
    (numRows, numCols) = scores.shape[2:4]
    rects = []
    confidences = []
    # loop over the number of rows
    for y in range(0, numRows):
        # extract the scores (probabilities), followed by the
        # geometrical data used to derive potential bounding box
        # coordinates that surround text
        scoresData = scores[0, 0, y]
        xData0 = geometry[0, 0, y]
        xData1 = geometry[0, 1, y]
        xData2 = geometry[0, 2, y]
        xData3 = geometry[0, 3, y]
        anglesData = geometry[0, 4, y]
        # loop over the number of columns
        for x in range(0, numCols):
            # if our score does not have sufficient probability,
            # ignore it
            if scoresData[x] < args.min_confidence:
                continue
            # compute the offset factor as our resulting feature
            # maps will be 4x smaller than the input image
            (offsetX, offsetY) = (x * 4.0, y * 4.0)
            # extract the rotation angle for the prediction and
            # then compute the sin and cosine
            angle = anglesData[x]
            cos = np.cos(angle)
            sin = np.sin(angle)
            # use the geometry volume to derive the width and height
            # of the bounding box
            h = xData0[x] + xData2[x]
            w = xData1[x] + xData3[x]
            # compute both the starting and ending (x, y)-coordinates
            # for the text prediction bounding box
            endX = int(offsetX + (cos * xData1[x]) + (sin * xData2[x]))
            endY = int(offsetY - (sin * xData1[x]) + (cos * xData2[x]))
            startX = int(endX - w)
            startY = int(endY - h)
            # add the bounding box coordinates and probability score
            # to our respective lists
            rects.append((startX, startY, endX, endY))
            confidences.append(scoresData[x])
    # return a tuple of the bounding boxes and associated confidences
    return (rects, confidences)


def load_and_run_network_model(image_obj: cv2.Mat, new_width: int, new_height: int):
    # define the two output layer names for the EAST detector model that
    # we are interested in -- the first is the output probabilities and the
    # second can be used to derive the bounding box coordinates of text
    layerNames = [
        "feature_fusion/Conv_7/Sigmoid",
        "feature_fusion/concat_3"]
    # load the pre-trained EAST text detector
    print("[INFO] loading EAST text detector...")
    net = cv2.dnn.readNet("frozen_east_text_detection.pb")

    # construct a blob from the image and then perform a forward pass of
    # the model to obtain the two output layer sets
    blob = cv2.dnn.blobFromImage(image_obj, 1.0, (new_width, new_height),
                                 (123.68, 116.78, 103.94), swapRB=True, crop=False)
    net.setInput(blob)
    (scores, geometry) = net.forward(layerNames)

    # decode the predictions, then  apply non-maxima suppression to
    # suppress weak, overlapping bounding boxes
    (rects, confidences) = decode_predictions(scores, geometry)
    boxes = non_max_suppression(np.array(rects), probs=confidences)

    return boxes


def extract_text_from_region_of_interest(startX: int, startY: int, endX: int, endY: int, configured_padding: int, original_image_obj: cv2.Mat, original_width: int, original_height: int, ratio_width: float, ratio_height: float):
    # scale the bounding box coordinates based on the respective ratios
    startX = int(startX * ratio_width)
    startY = int(startY * ratio_height)
    endX = int(endX * ratio_width)
    endY = int(endY * ratio_height)

    # in order to obtain a better OCR of the text we can potentially
    # apply a bit of padding surrounding the bounding box -- here we
    # are computing the deltas in both the x and y directions
    dX = int((endX - startX) * configured_padding)
    dY = int((endY - startY) * configured_padding)

    # apply padding to each side of the bounding box, respectively
    startX = max(0, startX - dX)
    startY = max(0, startY - dY)
    endX = min(original_width, endX + (dX * 2))
    endY = min(original_height, endY + (dY * 2))

    # extract the actual padded ROI
    roi = original_image_obj[startY:endY, startX:endX]

    # in order to apply Tesseract v4 to OCR text we must supply
    # (1) a language, (2) an OEM flag of 1, indicating that the we
    # wish to use the LSTM neural net model for OCR, and finally
    # (3) an OEM value, in this case, 7 which implies that we are
    # treating the ROI as a single line of text
    config = (f"-l eng --oem 1 --psm {str(args.psm)}")
    text = pytesseract.image_to_string(roi, config=config)

    # add the bounding box coordinates and OCR'd text to the list
    # of results

    truncatedStartY = args.truncate*int(startY/args.truncate)
    truncatedEndY = (1+args.truncate)*int(endY/args.truncate)
    result = ((startX, startY, endX, endY, truncatedStartY), text)
    return result


def extract_list_of_text_from_region_of_interest(boxes, configured_padding: int, original_image_obj: cv2.Mat, original_width: int, original_height: int, ratio_width: float, ratio_height: float):
    results = []
    # loop over the bounding boxes
    for (startX, startY, endX, endY) in boxes:
        result = extract_text_from_region_of_interest(
            startX, startY, endX, endY, configured_padding, original_image_obj, original_width, original_height, ratio_width, ratio_height)
        results.append(result)
    return results


def display_results(results: list, original_image_obj: cv2.Mat):
    # loop over the results
    for ((startX, startY, endX, endY, truncatedStartY), text) in results:
        # display the text OCR'd by Tesseract
        print("========")
        print(f"{startY},{startX}\n{text}")
        # strip out non-ASCII text so we can draw the text on the image
        # using OpenCV, then draw the text and a bounding box surrounding
        # the text region of the input image
        text = "".join([c if ord(c) < 128 else "" for c in text]).strip()

        output = original_image_obj.copy()
        cv2.rectangle(output, (startX, startY), (endX, endY),
                      (0, 0, 255), 2)
        cv2.putText(output, text, (startX, startY - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
        # show the output image
        cv2.imshow("Text Detection", output)
        cv2.waitKey(750)


def main(args: dict):
    image_obj = load_image(args.image)
    original_image_obj = image_obj.copy()
    (original_height, original_width) = image_obj.shape[:2]

    configured_new_width = args.width
    configured_new_height = args.height
    ratio_width = original_width / float(configured_new_width)
    ratio_height = original_height / float(configured_new_height)

    image_obj = resize_image(image_obj, original_width, original_height,
                             configured_new_width, configured_new_height)

    (new_height, new_width) = image_obj.shape[:2]

    boxes = load_and_run_network_model(image_obj, new_width, new_height)

    configured_padding = args.padding
    results = extract_list_of_text_from_region_of_interest(
        boxes, configured_padding, original_image_obj, original_width, original_height, ratio_width, ratio_height)

    # sort the results bounding box coordinates from top to bottom and left to right
    results = sorted(results, key=lambda r: r[0][4]*new_height+r[0][0])

    display_results(results, original_image_obj)


if __name__ == '__main__':
    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", type=str,
                    help="path to input image")
    ap.add_argument("-c", "--min-confidence", type=float, default=0.5,
                    help="minimum probability required to inspect a region")
    ap.add_argument("-w", "--width", type=int, default=320,
                    help="nearest multiple of 32 for resized width")
    ap.add_argument("-e", "--height", type=int, default=320,
                    help="nearest multiple of 32 for resized height")
    ap.add_argument("-p", "--padding", type=float, default=0.0,
                    help="amount of padding to add to each border of ROI")
    ap.add_argument("-m", "--psm", type=int, default=7,
                    help="Post segmentation mode for Tesseract")
    ap.add_argument("-t", "--truncate", type=int, default=7,
                    help="amount of truncate parameters for Y coords")
    args = ap.parse_args()
    main(args)
