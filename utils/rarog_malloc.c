#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <dlfcn.h>


void *rarog_malloc(void *ptr, size_t offset, size_t sz) {
    // fprintf(stderr, "malloc %p %ld\n", ptr+offset, sz);

    return ptr+offset;
}

void rarog_free(void *ptr_malloc, void *ptr) {
    // free(ptr);
    // fprintf(stderr, "free %p\n", ptr);
}