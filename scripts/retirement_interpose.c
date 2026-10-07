/* Private fixture only. Record native attempts even if candidate code catches EPERM. */
#include <sys/socket.h>
#include <sys/types.h>
#include <fcntl.h>
#include <unistd.h>
#include <errno.h>
#include <stdlib.h>
#include <stdio.h>
#include <libproc.h>
#include <mach-o/dyld.h>

/* Separate private diagnostics: identify the process that actually loaded this guard. */
__attribute__((constructor)) static void fixture_process_identity(void) {
    const char *path = getenv("RETIREMENT_PROCESS_IDENTITY");
    if (!path) return;
    const struct mach_header *header = NULL;
    for (uint32_t i = 0; i < _dyld_image_count(); i++) {
        const struct mach_header *image = _dyld_get_image_header(i);
        if (image && image->filetype == MH_EXECUTE) header = image;
    }
    char executable[PROC_PIDPATHINFO_MAXSIZE];
    int size = proc_pidpath(getpid(), executable, sizeof(executable));
    if (!header || size <= 0) _exit(92);
    int fd = open(path, O_CREAT | O_TRUNC | O_WRONLY, 0600);
    if (fd < 0) _exit(93);
    dprintf(fd, "{\"pid\":%d,\"cpu_type\":%u,\"cpu_subtype\":%u,\"executable_hex\":\"",
            getpid(), (unsigned)header->cputype, (unsigned)header->cpusubtype);
    for (int i = 0; i < size && executable[i]; i++) dprintf(fd, "%02x", (unsigned char)executable[i]);
    dprintf(fd, "\"}\n");
    close(fd);
}

static int refused(void) {
    const char *path = getenv("RETIREMENT_EVENTS");
    const char *phase = getenv("RETIREMENT_PHASE");
    if (!path) _exit(90);
    int fd = open(path, O_APPEND | O_CREAT | O_WRONLY, 0600);
    if (fd < 0) _exit(91);
    const char *label = phase && phase[0] == 'c' && phase[1] == 'o' ? "control" : "candidate";
    dprintf(fd, "{\"kind\":\"native.connect\",\"phase\":\"%s\"}\n", label);
    close(fd);
    errno = EPERM;
    return -1;
}

static int fixture_connect(int fd, const struct sockaddr *address, socklen_t length) {
    (void)fd; (void)address; (void)length;
    return refused();
}

static ssize_t fixture_sendto(int fd, const void *buffer, size_t size, int flags,
                              const struct sockaddr *address, socklen_t length) {
    (void)fd; (void)buffer; (void)size; (void)flags; (void)address; (void)length;
    return refused();
}

#define INTERPOSE(replacement, original) \
    __attribute__((used)) static struct { const void *new_fn; const void *old_fn; } \
    interpose_##original __attribute__((section("__DATA,__interpose"))) = \
    { (const void *)&replacement, (const void *)&original }

INTERPOSE(fixture_connect, connect);
INTERPOSE(fixture_sendto, sendto);
