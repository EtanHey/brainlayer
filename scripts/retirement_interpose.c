/* Private fixture only. Record native attempts even if candidate code catches EPERM. */
#include <sys/socket.h>
#include <sys/types.h>
#include <fcntl.h>
#include <unistd.h>
#include <errno.h>
#include <stdlib.h>
#include <stdio.h>

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
