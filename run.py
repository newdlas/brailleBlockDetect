import edgeDetect
import shapeDetect
import threading
import time

if __name__ == "__main__" :
    lock = threading.Lock()

    lock.acquire()
    t1 = threading.Thread(target=edgeDetect.run, daemon=True)
    t1.start()
    t1.join()
    lock.release()

    edge = edgeDetect.edgeGetter()
    img = edgeDetect.targetGetter()
    idx = edgeDetect.numGetter()
    print(idx)

    lock.acquire()
    shapeDetect.imgSetter(img, edge)
    t2 = threading.Thread(target=shapeDetect.run, args=idx, daemon=True)
    lock.release()
