#ifndef FYODOR_THEME_H
#define FYODOR_THEME_H
#include <stddef.h>
#include <stdint.h>
typedef enum {
    FYODOR_BACKGROUND, FYODOR_SURFACE, FYODOR_SURFACE_RAISED, FYODOR_BORDER,
    FYODOR_TEXT, FYODOR_TEXT_MUTED, FYODOR_TEXT_DISABLED, FYODOR_ACCENT,
    FYODOR_ACCENT_HOVER, FYODOR_ACCENT_TEXT, FYODOR_SUCCESS, FYODOR_WARNING,
    FYODOR_ERROR, FYODOR_INFO, FYODOR_FOCUS, FYODOR_SELECTION,
    FYODOR_USER_MESSAGE, FYODOR_ASSISTANT_MESSAGE, FYODOR_THEME_ROLE_COUNT
} fyodor_theme_role;
typedef struct { const char *id,*name; uint32_t colors[FYODOR_THEME_ROLE_COUNT]; } fyodor_theme;
typedef enum { FYODOR_COLOR_NONE, FYODOR_COLOR_16, FYODOR_COLOR_256, FYODOR_COLOR_TRUE } fyodor_color_mode;
size_t fyodor_theme_count(void);
const char *fyodor_theme_license(void);
const fyodor_theme *fyodor_theme_at(size_t index);
const fyodor_theme *fyodor_theme_find(const char *id);
/* ANSI SGR only, never user strings. Returns bytes or 0 on invalid/capacity. */
size_t fyodor_theme_sgr(const fyodor_theme *theme,fyodor_theme_role foreground,
                      fyodor_theme_role background,fyodor_color_mode mode,
                      char *output,size_t capacity);
#endif
