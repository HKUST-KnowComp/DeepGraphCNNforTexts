package ecs;

import java.util.concurrent.Executors;
import java.util.concurrent.ScheduledExecutorService;

/**
 * Created by LYP on 2016/11/24.
 */
public class CoreNLPService {
    static String pathPatch = "/storage1/lyp/InputFiles/";
    private static int threadNum = 50;
    private static int threadEnd = 50;
    private static int threadSta = 0;
    //bd62->20 80 60 1391700+463958=>9279*50
    //bd31->30 30 0
    //bd54->30 60 30
    public static void main(String[] args) {
//        String str = "java怎么把字符1串中的的汉字2取出来";
//        String reg = "[^0-9]";
//        str = str.replaceAll(reg, "");
//        System.out.println(str);
//        System.exit(-1);
        CoreNLPService coreNLPService = new CoreNLPService();
        coreNLPService.service();
    }

    public void service() {
        ScheduledExecutorService scheduler = Executors.newScheduledThreadPool(threadNum);
        int cnt = threadSta;
        while (cnt < threadEnd) {
            try {
                final int inner = cnt;
                final Runnable task = new Runnable() {
                    @Override
                    public void run() {
                        try {
                            System.out.println("process start!");
                            ProcessBuilder builder = new ProcessBuilder();
                            builder.redirectError(ProcessBuilder.Redirect.INHERIT);
                            builder.redirectOutput(ProcessBuilder.Redirect.INHERIT);

                            builder.environment().put("MAVEN_OPTS", "-Xmx6144m -XX:MaxPermSize=1536M");
                            String cmdLine = "mvn,exec:java,-Dexec.mainClass=ecs.TestCoreNLP,-Dexec.args=\"\"-i "
                                    + inner + " -c " + pathPatch + " -t 5" + "\"\"";
                            String[] cmdArray = cmdLine.split(",");
                            builder.command(cmdArray);

                            final Process process = builder.start();

                            Runtime.getRuntime().addShutdownHook(new Thread() {
                                @Override
                                public void run() {
                                    process.destroy();
                                }
                            });
                        }catch (Exception e) {
                            // TODO Auto-generated catch block
                            e.printStackTrace();
                        }
                    }
                };

                scheduler.submit(task);
                cnt++;
            }catch (Exception e) {
                // TODO Auto-generated catch block
                e.printStackTrace();
            }
        }
    }
}
