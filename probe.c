#include <stdio.h>
#include <stddef.h>
#include <string.h>
#include <errno.h>
#include <unistd.h>
#include <signal.h>
#include <libproc.h>
#include <sys/sysctl.h>
#include <sys/proc.h>
#include <sys/proc_info.h>
#include <sys/wait.h>

/* Candidate rule: exists(pid) && SHORTBSDINFO fails with ESRCH  => zombie */
static int exists(pid_t pid) {
    errno = 0;
    if (kill(pid, 0) == 0) return 1;
    return errno == EPERM;
}
static int rule_zombie(pid_t pid) {
    if (!exists(pid)) return 0;
    struct proc_bsdshortinfo si;
    memset(&si, 0, sizeof(si));
    errno = 0;
    int r = proc_pidinfo(pid, PROC_PIDT_SHORTBSDINFO, 0, &si, sizeof(si));
    return r != (int)sizeof(si) && errno == ESRCH;
}
static int truth_zombie(pid_t pid) {
    struct kinfo_proc kp; size_t len = sizeof(kp);
    int mib[4] = { CTL_KERN, KERN_PROC, KERN_PROC_PID, pid };
    memset(&kp, 0, sizeof(kp));
    if (sysctl(mib, 4, &kp, &len, NULL, 0) != 0) return -1;
    if (len == 0) return 0;
    return kp.kp_proc.p_stat == SZOMB;
}
static void row(const char *tag, pid_t pid) {
    printf("%-28s pid=%-7d rule=%d truth=%d exists=%d\n",
           tag, pid, rule_zombie(pid), truth_zombie(pid), exists(pid));
    fflush(stdout);
}

int main(void) {
    printf("sizeof(kinfo_proc)=%zu offsetof(kp_proc.p_stat)=%zu sizeof(p_stat)=%zu SZOMB=%d\n",
           sizeof(struct kinfo_proc),
           offsetof(struct kinfo_proc, kp_proc) + offsetof(struct extern_proc, p_stat),
           sizeof(((struct extern_proc *)0)->p_stat), SZOMB);

    /* short buffer behaviour: does sysctl fill a prefix or refuse? */
    unsigned char buf[64];
    size_t len = sizeof(buf);
    int mib[4] = { CTL_KERN, KERN_PROC, KERN_PROC_PID, getpid() };
    memset(buf, 0xAA, sizeof(buf));
    errno = 0;
    int sr = sysctl(mib, 4, buf, &len, NULL, 0);
    printf("short-buffer sysctl ret=%d len=%zu errno=%d(%s) byte36=%u\n",
           sr, len, errno, strerror(errno), buf[36]);

    row("self", getpid());
    row("launchd (pid 1, root)", 1);
    row("nonexistent (pid 999999)", 999999);

    pid_t pid = fork();
    if (pid == 0) { execl("/bin/sleep", "sleep", "60", (char *)NULL); _exit(127); }
    sleep(1);
    row("running child", pid);
    kill(pid, SIGKILL);
    for (int i = 0; i < 3; i++) { usleep(300000); row("zombie child", pid); }
    int st; waitpid(pid, &st, 0);
    row("reaped child", pid);
    return 0;
}
