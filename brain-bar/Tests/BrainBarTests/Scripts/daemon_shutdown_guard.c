#include <errno.h>
#include <fcntl.h>
#include <signal.h>
#include <spawn.h>
#include <sqlite3.h>
#include <stdarg.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#define INTERPOSE(replacement, original) \
 __attribute__((used)) static struct { const void *new_fn; const void *old_fn; } \
 interpose_##original __attribute__((section("__DATA,__interpose"))) = {(const void *)replacement, (const void *)original}

// Loaded only into the owned private daemon. Hold a real backup before its
// SQLite step; the RPC request and shutdown accounting remain production code.
static int held_step(sqlite3_stmt *statement) {
 const char *sql = sqlite3_sql(statement);
 const char *entered = getenv("BRAINBAR_SHUTDOWN_TEST_ENTERED");
 const char *release = getenv("BRAINBAR_SHUTDOWN_TEST_RELEASE");
 if (sql && strstr(sql, "VACUUM INTO") && entered && release) {
  int fd = open(entered, O_WRONLY | O_CREAT | O_TRUNC, 0600);
  if (fd >= 0) { write(fd, "entered", 7); close(fd); }
  while (access(release, F_OK) != 0) usleep(1000);
 }
 return sqlite3_step(statement);
}
static int private_open(const char *path, int flags, ...) {
 if (strcmp(path, "/tmp/brainbar-debug.log") == 0) { errno = ENOENT; return -1; }
 mode_t mode = 0;
 if (flags & O_CREAT) { va_list args; va_start(args, flags); mode = (mode_t)va_arg(args, int); va_end(args); }
 return open(path, flags, mode);
}
// A private test process must never signal or launch a production peer.
static int private_kill(pid_t pid, int sig) {
 if (pid != getpid()) { errno = EPERM; return -1; }
 return kill(pid, sig);
}
static int private_spawn(pid_t *pid, const char *path, const posix_spawn_file_actions_t *actions,
 const posix_spawnattr_t *attributes, char *const argv[], char *const env[]) {
 if (strstr(path, "launchctl") || strcmp(path, "/usr/bin/open") == 0) return EPERM;
 return posix_spawn(pid, path, actions, attributes, argv, env);
}
INTERPOSE(held_step, sqlite3_step);
INTERPOSE(private_open, open);
INTERPOSE(private_kill, kill);
INTERPOSE(private_spawn, posix_spawn);
