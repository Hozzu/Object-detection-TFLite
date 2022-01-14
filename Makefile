LIB_INCS = -I include/
LIB_PATH = -L lib/
LDFLAGS = -ltensorflowlite -ltensorflowlite_gpu_delegate -ltensorflowlite_hexagon_delegate -lais_client -lfastcvopt -ljson-c -ljpeg -lqcarcam_client -lopencv_core -lopencv_imgproc
CFLAGS = -O2

all:
	$(CXX) $(CFLAGS) $(LIB_INCS) $(LIB_PATH) src/run_image.cpp src/run_camera.cpp src/main.cpp -o pkshin_detect $(LDFLAGS)
clean:
	rm -f pkshin_detect
