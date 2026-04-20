#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <dlfcn.h>


void *rarog_malloc(void *ptr, size_t offset) {
    fprintf(stderr, "malloc %p %d\n", ptr, offset);

    return ptr+offset;
}

void rarog_free(void *ptr_malloc, void *ptr) {
    // free(ptr);
    fprintf(stderr, "free %p\n", ptr);
}