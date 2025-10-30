import argparse
import os
import time
import cv2
import mediapipe as mp
import numpy as np
import pickle
from collections import deque

# optional sklearn import (used during training/prediction)
try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report
except Exception:
    RandomForestClassifier = None


def parse_args():
    parser = argparse.ArgumentParser(description="Webcam hand detection + highlight using MediaPipe")
    parser.add_argument("--camera", type=int, default=0, help="Camera device index (default: 0)")
    parser.add_argument("--max_num_hands", type=int, default=2, help="Maximum number of hands to detect")
    parser.add_argument("--model_complexity", type=int, default=1, choices=[0, 1, 2], help="Model complexity (0,1,2)")
    parser.add_argument("--min_detection_confidence", type=float, default=0.5)
    parser.add_argument("--min_tracking_confidence", type=float, default=0.5)
    parser.add_argument("--record", type=str, default=None, help="Record samples for a given LABEL (single char/word). Press 's' to save a sample")
    parser.add_argument("--train", action="store_true", help="Train a classifier from recorded samples (dataset.csv -> model.pkl)")
    parser.add_argument("--model_path", type=str, default="model.pkl", help="Path to save/load trained model")
    parser.add_argument("--two_hands", action="store_true", help="Use concatenated left+right hand features (2x63 floats) for recording/training/prediction")
    parser.add_argument("--demo", type=str, help="Demo mode: provide text to cycle through (e.g., 'hello my name is judah')")
    parser.add_argument("--name", type=str, default="JUDAH",help="will but your name here when using the name sign")
    return parser.parse_args()


def main():
    args = parse_args()

    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils

    cap = cv2.VideoCapture(args.camera)
    
    # Set resolution to 1920x1080
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    
    if not cap.isOpened():
        print(f"Cannot open camera {args.camera}")
        return

    # Get actual resolution (camera might not support exactly 1920x1080)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Camera resolution: {width}x{height}")

    with mp_hands.Hands(
        max_num_hands=args.max_num_hands,
        model_complexity=args.model_complexity,
        min_detection_confidence=args.min_detection_confidence,
        min_tracking_confidence=args.min_tracking_confidence,
    ) as hands:

        # Clear predictions.txt at startup
        with open('predictions.txt', 'w') as f:
            f.write('')

        # If training requested, we'll collect features and train after the loop
        prev_time = 0
        # Prediction smoothing
        pred_queue = deque(maxlen=8)
        smoothed_label = None
        current_text = ""
        
        # Prediction timing tracking
        last_prediction = None
        prediction_start_time = None
        prediction_duration = 1.0  # seconds required for stable prediction
        global pred_words
        pred_words = []

        # Demo mode setup
        demo_words = []
        demo_word_index = 0
        demo_start_time = time.time()
        demo_word_display_time = 1.0  # seconds per word
        demo = False
        if args.demo:
            demo_words_caps = args.demo.upper()
            demo_words = demo_words_caps.strip().split()
            
            print(f"Demo mode: Will cycle through words: {demo_words}")
        
        #Pred bool
        pred_bool = True

        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame")
                break

            # Flip for a selfie-view and convert to RGB
            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            results = hands.process(rgb)
            rgb.flags.writeable = True

            overlay = frame.copy()

            # Build per-frame features for single or two-hand mode
            current_feature = None
            # helper: convert a MediaPipe hand_landmarks to a 63-float vector
            def landmarks_to_vector(hl):
                v = []
                for lm in hl.landmark:
                    v.extend([lm.x, lm.y, lm.z])
                return v

            left_vec = None
            right_vec = None
            single_hand_vec = None
            if results.multi_hand_landmarks:
                # results.multi_handedness aligns with multi_hand_landmarks by index
                for i, hand_landmarks in enumerate(results.multi_hand_landmarks):
                    # Draw landmarks as before
                    pts = []
                    for lm in hand_landmarks.landmark:
                        x_px = int(lm.x * width)
                        y_px = int(lm.y * height)
                        pts.append((x_px, y_px))

                    pts_np = np.array(pts, dtype=np.int32)

                    ## add outline fill
                    # try:
                    #     hull = cv2.convexHull(pts_np)
                    #     cv2.fillConvexPoly(overlay, hull, (0, 255, 0))
                    # except Exception:
                    #     cv2.fillPoly(overlay, [pts_np], (0, 255, 0))

                    mp_drawing.draw_landmarks(
                        frame,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2, circle_radius=2),
                        mp_drawing.DrawingSpec(color=(0, 0, 0), thickness=2),
                    )

                    vec = landmarks_to_vector(hand_landmarks)
                    single_hand_vec = vec  # last-hand fallback (keeps previous behavior)

                    # Determine handedness label for this hand if available
                    label = None
                    try:
                        if results.multi_handedness and len(results.multi_handedness) > i:
                            classification = results.multi_handedness[i]
                            # classification may have .classification[0].label or .label
                            if hasattr(classification, 'classification') and len(classification.classification) > 0 and hasattr(classification.classification[0], 'label'):
                                label = classification.classification[0].label
                            elif hasattr(classification, 'label'):
                                label = classification.label
                    except Exception:
                        label = None

                    if label:
                        l = label.lower()
                        if 'left' in l:
                            left_vec = vec
                        elif 'right' in l:
                            right_vec = vec

                # Build combined feature when two_hands enabled
                if args.two_hands:
                    zeros63 = [0.0] * 63
                    left_final = left_vec if left_vec is not None else zeros63
                    right_final = right_vec if right_vec is not None else zeros63
                    current_feature = left_final + right_final
                else:
                    current_feature = single_hand_vec

            # Recording mode: show prompt
            if args.record:
                mode_note = "(two-hands)" if args.two_hands else "(single-hand)"
                cv2.putText(frame, f"RECORD MODE {mode_note}: press 's' to save sample for '{args.record}'", (10, height - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            # Prediction mode: if model exists, predict and smooth
            def pred():
                nonlocal last_prediction, prediction_start_time
                model = None
                if os.path.exists(args.model_path):
                    try:
                        with open(args.model_path, "rb") as f:
                            model = pickle.load(f)
                    except Exception:
                        model = None

                if model is not None and current_feature is not None:
                    try:
                        pred = model.predict([current_feature])[0]
                        pred_queue.append(pred)
                        if len(pred_queue) == pred_queue.maxlen:
                            smoothed_label = max(set(pred_queue), key=pred_queue.count)
                        else:
                            smoothed_label = pred
                        if pred=="VAR_NAME" or smoothed_label == "VAR_NAME":
                            smoothed_label = args.name
                        
                        # Track prediction timing
                        current_time = time.time()
                        if smoothed_label != last_prediction:
                            # Prediction changed, reset timing
                            last_prediction = smoothed_label
                            prediction_start_time = current_time
                        elif prediction_start_time is not None:
                            # Check if prediction has been stable for required duration
                            if current_time - prediction_start_time >= prediction_duration:
                                # Write to file if we haven't already written this prediction
                                
                                with open('predictions.txt', 'a') as f:
                                    f.write(f"{smoothed_label}\n")
                                
                                pred_words.append(smoothed_label)
                                print(f"Added {smoothed_label} to predictions")
                                if smoothed_label == "CLEAR":
                                    pred_words.clear()
                                # Reset timing to prevent multiple writes of same prediction
                                prediction_start_time = None
                        
                        cv2.putText(frame, f"Pred: {smoothed_label}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
                        cv2.putText(frame, f" {' '.join(pred_words)}", (10, 160), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
                    except Exception as e:
                        print(f"Prediction error: {e}")
                        pass
            if demo_words == []:
                if pred_bool == True:
                    pred()
            # Blend overlay with frame (alpha)
            alpha = 0.25
            cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
            key = cv2.waitKey(1) & 0xFF
           
            if key == ord('d'):
                demo = True
            # Demo mode word display
            if demo == True:
                if args.demo and demo_words:
                    current_time = time.time()
                    elapsed = current_time - demo_start_time

                    # Display current word
                    if demo_word_index == len(demo_words):
                        demo_words= []
                        

                    # Check if it's time to move to next word
                    if elapsed >= demo_word_display_time:

                        demo_word_index+=1
                        demo_start_time = current_time


                    if demo_word_index < len(demo_words):
                        current_word = demo_words[demo_word_index]
                    # Calculate text size to center it
                    cv2.putText(frame, f"Pred: {current_word}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

            cv2.imshow("MediaPipe Hands - Highlight", frame)

            
            if key == 27 or key == ord('q'):
                break

            if key == ord('s') and args.record and current_feature is not None:
                # save current_feature to dataset.csv
                os.makedirs(os.path.dirname("dataset.csv") or ".", exist_ok=True)
                row = ",".join([str(args.record)] + [f"{v:.6f}" for v in current_feature])
                with open('dataset.csv', 'a') as fh:
                    fh.write(row + "\n")
                print(f"Saved sample for '{args.record}' (total features len={len(current_feature)})")

            if key == ord(' ') and smoothed_label:
                current_text += str(smoothed_label)
                print(f"Text: {current_text}")

            if key == ord('c'):
                current_text = ""
                demo_words = []
                pred_bool = False

            if key == ord('p'):
                current_text = ""
                pred_bool = True
                if pred_bool == True:
                    pred()


            if key == ord('b'):
                current_text = current_text[:-1]

            # If training requested and user pressed 't', break loop and train
            if args.train and key == ord('t'):
                print("Starting training...")
                break

    cap.release()
    cv2.destroyAllWindows()

    # If training flag passed, train a classifier from dataset.csv
    if args.train:
        data_path = 'dataset.csv'
        if not os.path.exists(data_path):
            print(f"No dataset found at {data_path}. Record samples first using --record LABEL and pressing 's' to save.")
            return

        # Load dataset: rows label,f1,f2,...
        labels = []
        features = []
        print("\nLoading dataset...")
        start_time = time.time()
        with open(data_path, 'r') as fh:
            for line in fh:
                parts = line.strip().split(',')
                if len(parts) < 2:
                    continue
                labels.append(parts[0])
                feats = list(map(float, parts[1:]))
                features.append(feats)
        
        load_time = time.time() - start_time
        # Show dataset stats
        from collections import Counter
        class_counts = Counter(labels)
        print(f"\nDataset loaded in {load_time:.2f}s:")
        print(f"Total samples: {len(labels)}")
        print("Samples per class:")
        for label, count in sorted(class_counts.items()):
            print(f"  {label}: {count}")

        # Validate consistent feature length
        lengths = set(len(f) for f in features)
        if len(lengths) != 1:
            print(f"Inconsistent feature lengths in {data_path}: {lengths}. Clean or re-record data so all rows have the same feature count.")
            return

        X = np.array(features)
        y = np.array(labels)

        if RandomForestClassifier is None:
            print("scikit-learn is not installed in this environment. Install scikit-learn to enable training.")
            return

        print("\nTraining RandomForest classifier...")
        train_start = time.time()
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        clf = RandomForestClassifier(n_estimators=100, random_state=42)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        train_time = time.time() - train_start
        
        print(f"\nTraining completed in {train_time:.2f}s")
        print("\nClassification report:")
        print(classification_report(y_test, y_pred))

        # Save model
        save_start = time.time()
        with open(args.model_path, 'wb') as f:
            pickle.dump(clf, f)
        save_time = time.time() - save_start
        print(f"\nModel saved to {args.model_path} in {save_time:.2f}s")
        print(f"Total time: {time.time() - start_time:.2f}s")


if __name__ == "__main__":
    main()
