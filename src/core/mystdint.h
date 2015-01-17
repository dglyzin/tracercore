#ifndef _mystdint
#define _mystdint
typedef signed char        int8_t;
typedef short              int16_t;
typedef int                int32_t;
typedef unsigned char      uint8_t;
typedef unsigned short     uint16_t;
typedef unsigned int       uint32_t;



#ifdef WIN32
#define EXPORT_XX __declspec(dllexport)
typedef long long          int64_t;
typedef unsigned long long uint64_t;


#else
#define EXPORT_XX
#endif

#define TEMP_GRAPH_LEN 50
#define ROOT_NODE 0
#define BINARY_FILE_VERSION_MAJ 2
#define BINARY_FILE_VERSION_MIN 0

enum block_side_t {XSTART = 0, XEND = 1, YSTART = 2, YEND = 3, ZSTART = 4, ZEND = 5};
enum axis_dir_t {AXISX = 0, AXISY = 1, AXISZ = 2};
enum body_type_t {BT_SOIL = 0, BT_CAVERN = 1};
enum border_type_t {BRT_BOUND = 0, BRT_IC = 1};

#endif
