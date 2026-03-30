#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <dlfcn.h>


void *instrumented_malloc(size_t size) {
    void *ptr = malloc(size);
    printf("malloc %p %d\n", ptr, (int)size);

    return ptr;
}

// Optionally also wrap free, calloc, etc.
void instrumented_free(void *ptr) {
    free(ptr);
    printf("free %p\n", ptr);
}