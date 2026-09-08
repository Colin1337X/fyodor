#ifndef NYA_H
#define NYA_H

/* The public engine version is kept in one place for logs and API replies. */
#define NYA_VERSION_MAJOR 0
#define NYA_VERSION_MINOR 3
#define NYA_VERSION_PATCH 0

/* Every JSON reply uses this protocol version. */
#define NYA_API_VERSION 1

/* A successful operation always returns zero. */
#define NYA_OK 0

/* A failed operation always returns a negative value. */
#define NYA_ERROR (-1)

#endif
