#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <unistd.h>
#include <signal.h>
#include <libproc.h>
#include <sys/sysctl.h>
#include <sys/proc.h>
#include <sys/proc_info.h>

static void probe(const char *tag, pid_t pid) {
    printf("--- %s pid=%d ---\n", tag, pid);

    struct proc_bsdshortinfo si;
    memset(&si, 0, sizeof(si));
    errno = 0;
    int r = proc_pidinfo(pid, PROC_PIDT_SHORTBSDINFO, 0, &si, sizeof(si));
    printf("SHORTBSDINFO ret=%d (want %d) errno=%d(%s) pbsi_status=%u\n",
           r, (int)sizeof(si), errno, strerror(errno), si.pbsi_status);

    struct proc_bsdinfo bi;
    memset(&bi, 0, sizeof(bi));
    errno = 0;
    r = proc_pidinfo(pid, PROC_PIDTBSDINFO, 0, &bi, sizeof(bi));
    printf("PIDTBSDINFO  ret=%d (want %d) errno=%d(%s) pbi_status=%u\n",
           r, (int)sizeof(bi), errno, strerror(errno), bi.pbi_status);

    struct kinfo_proc kp;
    size_t len = sizeof(kp);
    int mib[4] = { CTL_KERN, KERN_PROC, KERN_PROC_PID, pid };
    memset(&kp, 0, sizeof(kp));
    errno = 0;
    int sr = sysctl(mib, 4, &kp, &len, NULL, 0);
    printf("sysctl ret=%d len=%zu errno=%d(%s) p_stat=%d (SZOMB=%d) p_pid=%d\n",
           sr, len, errno, strerror(errno), (int)kp.kp_proc.p_stat, SZOMB, kp.kp_proc.p_pid);

    errno = 0;
    printf("kill(pid,0)=%d errno=%d(%s)\n", kill(pid, 0), errno, strerror(errno));

    char buf[PROC_PIDPATHINFO_MAXSIZE];
    errno = 0;
    r = proc_pidpath(pid, buf, sizeof(buf));
    printf("proc_pidpath ret=%d errno=%d(%s)\n", r, errno, strerror(errno));
    fflush(stdout);
}

int main(void) {
    printf("SIDL=%d SRUN=%d SSLEEP=%d SSTOP=%d SZOMB=%d\n", SIDL, SRUN, SSLEEP, SSTOP, SZOMB);
    pid_t pid = fork();
    if (pid == 0) { execl("/bin/sleep", "sleep", "60", (char *)NULL); _exit(127); }
    sleep(1);
    probe("running child", pid);
    kill(pid, SIGKILL);
    for (int i = 0; i < 5; i++) {
        usleep(200000);
        char tag[64];
        snprintf(tag, sizeof(tag), "after kill, no wait, t=%dms", (i + 1) * 200);
        probe(tag, pid);
    }
    int st; waitpid(pid, &st, 0);
    probe("after reap", pid);
    return 0;
}
