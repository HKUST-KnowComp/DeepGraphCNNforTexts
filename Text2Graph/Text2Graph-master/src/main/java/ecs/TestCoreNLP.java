package ecs;

import ecs.ansj.vec.Word2VEC;
import edu.stanford.nlp.ling.CoreAnnotations.*;
import edu.stanford.nlp.ling.CoreLabel;
import edu.stanford.nlp.pipeline.Annotation;
import edu.stanford.nlp.pipeline.StanfordCoreNLP;
import edu.stanford.nlp.semgraph.SemanticGraph;
import edu.stanford.nlp.semgraph.SemanticGraphCoreAnnotations;
import edu.stanford.nlp.semgraph.SemanticGraphEdge;
import edu.stanford.nlp.util.CoreMap;
import org.apache.commons.cli.*;
import org.apache.commons.io.FileUtils;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.*;

/**
 * Created by LYP on 2016/11/17.
 */
public class TestCoreNLP {
//    static String pathplus = "D:/DataSet/Song teacher/rcv1v2/ReutersCorpusVolume1/";
    private static int inner = 1;
    private static int range = 9279;//925*(0~24)=>23149 21526*(0~24)=>Lshtc 1855658=>nyt 74226*25
    //bd40->0~4 bd62->5~34 (53108*45)
    //5*53108 = 20*13277
    //23195*80
    //bd31 0~29 bd54 30~59 bd62 60~79
    //实际上 = 1852426
    private Word2VEC vec = new Word2VEC();
    static String pathPatch = "";
    static float unit = 1.1f;
    private HashMap<String,Integer> printWord2Index = new HashMap<>();
    private int printCnt = 0;
    private static int testKey = 0;
    private static int threadEnd = 49;

    private TestCoreNLP(){
        vec.setTopNSize(16);
        try {
//            vec.loadJavaModel("D:/DataSet/wiki/javaSkipRouters50");
//            vec.loadJavaModel(pathPatch+"javaSkip_Reuter50");
//            vec.loadJavaModel("D:/DataSet/Song teacher/LSHTC/v3/javaSkip_LSHTCAll50");
            vec.loadJavaModel(pathPatch+"NYT/javaSkip_NYT50");
//            vec.loadJavaModel("D:/123/javaSkip_WikiStem50");// put on bd -> change need
            System.out.println("wordMap size = "+vec.getWordMap().size());;
        } catch (IOException e) {
            System.out.println("word2vec load failed");
            e.printStackTrace();
        }
    }

    public static void main(String[] args) throws IOException, ParseException {
//        System.out.println(new TestCoreNLP().processNYT(new File("C:/Users/LYP/Desktop/02/02/06/0010648.xml")));System.exit(-1);

//        HashSet<String> hashSet = new HashSet<>();
//        List<String> fileList = FileUtils.readLines(new File("D:/DataSet/Song teacher/LSHTC/v3/wikipediaMediumOriginalLSHTCv3-train.txt"));
//        for(int i = 2;i < fileList.size();i += 5){
//            String label = fileList.get(i);
//            label = label.substring(label.indexOf('>')+1,label.lastIndexOf('<'));
//            String[] labels = label.split(" ");
//            for(String str:labels){
//                hashSet.add(str);
//            }
//        }
//        System.out.println(hashSet.size());
//        List<String> list = new ArrayList<>();
//        int i = 0;
//        for(Object object:hashSet.toArray()){
//            list.add(object + " " + i++);
//        }
//        FileUtils.writeLines(new File("D:/DataSet/Song teacher/LSHTC/v3/remap.txt"), list);
////        System.out.println(FileUtils.readLines(new File("D:/DataSet/Song teacher/LSHTC/v3/wikipediaMediumOriginalLSHTCv3-train.txt")).size());
//        System.exit(-1);
        Options opts = new Options();
        opts.addOption("i", "inner", true, "");
        opts.addOption("c", "pathPatch", true, "");
        opts.addOption("t", "testKey", true, "");
        CommandLineParser parser = new GnuParser();
        CommandLine commandLine = parser.parse(opts, args, true);
        inner = Integer.valueOf(commandLine.getOptionValue("i", "1"));
        pathPatch = commandLine.getOptionValue("c", "D:/DataSet/Song teacher/rcv1v2/ReutersCorpusVolume1/");
        testKey = Integer.valueOf(commandLine.getOptionValue("t", "3"));
//        String str = "java怎么把字符1串中的的汉字2取出来";
//        String reg = "[^0-9]";
//        str = str.replaceAll(reg, "");
//        System.out.println(str);
//        System.exit(-1);

        TestCoreNLP testCoreNLP = new TestCoreNLP();//testCoreNLP.printVecFirst("/storage1/lyp/InputFiles/NYT/index2vec");System.exit(-1);
        if(testKey == 1)
            testCoreNLP.test();
        else if(testKey == 2)
            testCoreNLP.appendGraphWithLabel();
        else if(testKey == 3)
            testCoreNLP.LSHTC();
        else if(testKey == 4)
            testCoreNLP.appendGraph();
        else if(testKey == 5)
            testCoreNLP.NYTimes();
        else if(testKey == 6)
            testCoreNLP.NYTimesLabel();
        else if(testKey == 7)
            testCoreNLP.heyuRequest();
        else if(testKey == 8)
            testCoreNLP.appendGraphNYT();
        else if(testKey == 9)
            testCoreNLP.appendGraphNYTWithHehe();
        else
            testCoreNLP.Kim();
    }

    private int appendGraph() throws IOException {
        String rootPath = pathPatch+"LSHTC/GraphAll/";
        List<String> list = FileUtils.readLines(new File(pathPatch + "LSHTC/remap.txt"));
        HashMap<String,String> label2index = new HashMap<>();
        for(String labelTuple:list){
            int index = labelTuple.indexOf(" ");
            label2index.put(labelTuple.substring(0,index),labelTuple.substring(1 + index));
        }
        list.clear();
        for(int i = 1;i <= 456886;++i){
            list.addAll(FileUtils.readLines(new File(rootPath + i + ".graph")));
            String labels = list.get(list.size() - 1), append = "";
            String[] strList = labels.split(" ");
            for(String tmp: strList){
                append += label2index.get(tmp) + " ";
            }
//            System.out.println(append);
            list.set(list.size() - 1,append);
//            list.add(FileUtils.readFileToString(new File(rootPath + i + ".graph")));
        }
        FileUtils.writeLines(new File(pathPatch+"data_LSHTC_All/train_graphs"), list);
        list.clear();
        for(int i = 456887;i <= 538148;++i){
            list.add(FileUtils.readFileToString(new File(rootPath + i + ".graph")));
        }
        FileUtils.writeLines(new File(pathPatch+"data_LSHTC_All/test_graphs"), list , "");
        return 0;
    }

    private void appendGraphNYTWithHehe() throws IOException {
        List<String> list = new ArrayList<>() , list_hehe = new ArrayList<>()  , listHier = new ArrayList<>() , listHier_hehe = new ArrayList<>();
        list = FileUtils.readLines(new File(pathPatch + "NYT/labelMap.txt"));
        list_hehe = FileUtils.readLines(new File(pathPatch + "NYT/labelMap_hehe.txt"));
        HashMap<String,String> map = new HashMap<>() ;
        for(int i = 0;i < list.size();++i){
            String s = list.get(i).split(" ")[0] , s_hehe = list_hehe.get(i).split(" ")[0];
            map.put(s,s_hehe);
        }
        listHier = FileUtils.readLines(new File(pathPatch + "NYT/labelInherit.txt"));
        for(int i = 0;i < listHier.size();++i){
            if(listHier.get(i).isEmpty()){
                listHier_hehe.add(listHier.get(i));
                continue;
            }
            String[] s = listHier.get(i).split(" ");
            String str = "";
            for(int j = 0;j < s.length;++j){
                str += map.get(s[j]) + " ";
            }
            listHier_hehe.add(str);
        }
        FileUtils.writeLines(new File(pathPatch + "NYT/labelInherit_hehe.txt"),listHier_hehe);
    }

    private int appendGraphNYT() throws IOException {
        String rootPath = pathPatch+"NYT/Graph/";
        String rootLabelPath = pathPatch + "NYT/Label_hehe/";
        String rootXmlPath = pathPatch + "NYT_annotated_corpus/data_all/";
        File root = new File(rootPath);
        File[] files = root.listFiles();

        List<String> list = new ArrayList<>() , list1 = new ArrayList<>()  , listXml = new ArrayList<>();
        list = FileUtils.readLines(new File(pathPatch + "NYT/remap.txt"));
        HashMap<String,String> map = new HashMap<>() , deleteMap = new HashMap<>();
        for(String str:list){
            String[] s = str.split(" ");
            map.put(s[0],s[1]);
        }
        list.clear();

        list = FileUtils.readLines(new File(pathPatch + "NYT/data.delete"));
        for(String str:list){
            deleteMap.put(str,"");
        }
        list.clear();

        for(File file:files){
            if(!deleteMap.containsKey(file.getName()))
                list.add(file.getName());
        }
        int size = list.size() , trainSize = size - size / 10;
        System.out.println("size:" + size + " trainSize:" + trainSize + " testSize:" + (size - trainSize));

        Collections.shuffle(list);//乱序

        root = new File(rootXmlPath);
        files = root.listFiles();
        for(int inn = 0;inn < files.length;++inn){
            listXml.add(files[inn].getName());
        }

        FileUtils.writeLines(new File(pathPatch+"NYT/train_list"), list.subList(0,trainSize));
        FileUtils.writeLines(new File(pathPatch+"NYT/test_list"), list.subList(trainSize,size));

        FileWriter writer = new FileWriter(pathPatch+"NYT/data.train");
        BufferedWriter bw = new BufferedWriter(writer);
        for(int i = 0;i < trainSize;++i){
//            System.out.println(list.get(i));
            int inn = Integer.valueOf(list.get(i).substring(0,list.get(i).indexOf('.')));
            String labelPath = inn + ".label";
//            String xmlPath = listXml.get(inn);
            String labelSplit = FileUtils.readFileToString(new File(rootLabelPath + labelPath));

            List<String> tmp = FileUtils.readLines(new File(rootPath + list.get(i)));

            for(int j = 0;j < 4;++j){
                bw.write(tmp.get(j) + "\r\n");
            }
            String[] s = labelSplit.split(" ");
            for(int j = 0;j < s.length;++j)
                bw.write(map.get(s[j]) + " ");
            bw.write("\r\n");
        }
        bw.close();
        writer.close();

        writer = new FileWriter(pathPatch+"NYT/data.test");
        bw = new BufferedWriter(writer);
        for(int i = trainSize;i < size;++i){
            int inn = Integer.valueOf(list.get(i).substring(0,list.get(i).indexOf('.')));
//            String xmlPath = listXml.get(inn);
            String labelPath = inn + ".label";

            List<String> tmp = FileUtils.readLines(new File(rootPath + list.get(i)));
            for(int j = 0;j < 4;++j){
                bw.write(tmp.get(j) + "\r\n");
            }
            String labelSplit = FileUtils.readFileToString(new File(rootLabelPath + labelPath));
            String[] s = labelSplit.split(" ");
            for(int j = 0;j < s.length;++j)
                bw.write(map.get(s[j]) + " ");
            bw.write("\r\n");
        }
        bw.close();
        writer.close();
//        FileUtils.writeLines(new File(pathPatch+"NYT/data.train"), list1.subList(0,trainSize));
//        FileUtils.writeLines(new File(pathPatch+"NYT/data.test"), list1.subList(trainSize,size));
        return 0;
    }

    private int deleteGraphNYT() throws IOException {
        String rootPath = pathPatch+"NYT/Graph/";
        String rootLabelPath = pathPatch + "NYT/Label_hehe/";
        String rootXmlPath = pathPatch + "NYT_annotated_corpus/data_all/";
        File root = new File(rootPath);
        File[] files = root.listFiles();
        int size = files.length , trainSize = 1667184;
        System.out.println(size);
        List<String> list = new ArrayList<>() , list1 = new ArrayList<>() , listDelete = new ArrayList<>() , listXml = new ArrayList<>();
        list = FileUtils.readLines(new File(pathPatch + "NYT/remap.txt"));
        HashMap<String,String> map = new HashMap<>();
        for(String str:list){
            String[] s = str.split(" ");
            map.put(s[0],s[1]);
        }
        list.clear();

        for(File file:files){
            list.add(file.getName());
        }
        Collections.shuffle(list);//乱序

        root = new File(rootXmlPath);
        files = root.listFiles();
        for(int inn = 0;inn < files.length;++inn){
            listXml.add(files[inn].getName());
        }

        FileUtils.writeLines(new File(pathPatch+"NYT/train_list"), list.subList(0,trainSize));
        FileUtils.writeLines(new File(pathPatch+"NYT/test_list"), list.subList(trainSize,size));

        FileWriter writer = new FileWriter(pathPatch+"NYT/data.train1");
        BufferedWriter bw = new BufferedWriter(writer);
        for(int i = 0;i < trainSize;++i){
//            System.out.println(list.get(i));
            int inn = Integer.valueOf(list.get(i).substring(0,list.get(i).indexOf('.')));
            String labelPath = inn + ".label";
            String xmlPath = listXml.get(inn);
            String labelSplit = FileUtils.readFileToString(new File(rootLabelPath + labelPath));
            if(labelSplit.trim().isEmpty()){
                listDelete.add(list.get(i));
                continue;
//                String ttmp = FileUtils.readFileToString(new File(rootXmlPath + xmlPath));
//                System.out.println(ttmp);
            }

            List<String> tmp = FileUtils.readLines(new File(rootPath + list.get(i)));
            if(tmp.size() != 5 || Integer.valueOf(tmp.get(0)) < 2){
                listDelete.add(list.get(i));
                continue;
            }
//            for(int j = 0;j < 4;++j){
//                bw.write(tmp.get(j) + "\r\n");
//            }
//            String[] s = labelSplit.split(" ");
//            for(int j = 0;j < s.length;++j)
//                bw.write(map.get(s[j]) + " ");
//            bw.write("\r\n");
        }
        bw.close();
        writer.close();

        writer = new FileWriter(pathPatch+"NYT/data.test1");
        bw = new BufferedWriter(writer);
        for(int i = trainSize;i < size;++i){
            int inn = Integer.valueOf(list.get(i).substring(0,list.get(i).indexOf('.')));
            String xmlPath = listXml.get(inn);
            String labelPath = inn + ".label";
            String labelSplit = FileUtils.readFileToString(new File(rootLabelPath + labelPath));
            if(labelSplit.trim().isEmpty()){
                listDelete.add(list.get(i));
                continue;
//                String ttmp = FileUtils.readFileToString(new File(rootXmlPath + xmlPath));
//                System.out.println(ttmp);
            }
            List<String> tmp = FileUtils.readLines(new File(rootPath + list.get(i)));
            if(tmp.size() != 5 || Integer.valueOf(tmp.get(0)) < 2){
                listDelete.add(list.get(i));
                continue;
            }
//            for(int j = 0;j < 4;++j){
//                bw.write(tmp.get(j) + "\r\n");
//            }
//            String labelSplit = FileUtils.readFileToString(new File(rootLabelPath + labelPath));
//            String[] s = labelSplit.split(" ");
//            for(int j = 0;j < s.length;++j)
//                bw.write(map.get(s[j]) + " ");
//            bw.write("\r\n");
        }
        bw.close();
        writer.close();

        FileUtils.writeLines(new File(pathPatch+"NYT/data.delete"), listDelete);

//        FileUtils.writeLines(new File(pathPatch+"NYT/data.train"), list1.subList(0,trainSize));
//        FileUtils.writeLines(new File(pathPatch+"NYT/data.test"), list1.subList(trainSize,size));
        return 0;
    }

    private int appendGraphWithLabel() throws IOException {
        List<String> list = FileUtils.readLines(new File(pathPatch+"heyu/rcv1.topics.txt"));
        HashMap<String,Integer> cat2index = new HashMap<>();
        for(int i = 0;i < list.size();i++){
            cat2index.put(list.get(i),i);
        }
        list = FileUtils.readLines(new File(pathPatch+"heyu/rcv1v2-ids-23149.dat"));
        HashSet<String> IdSet = new HashSet<>();
        for(int i = 0;i < list.size();i++){
            IdSet.add(list.get(i));
        }
        list = FileUtils.readLines(new File(pathPatch+"heyu/rcv1-v2.topics.qrels"));
        String lastid = "";
        String rootPath = pathPatch+"RouterGraph/";
        FileWriter writer = new FileWriter(pathPatch+"data_Router/test_graphs");
        BufferedWriter bw = new BufferedWriter(writer);

//        List<String> existList = new ArrayList<>();
        List<Integer> patch = new ArrayList<>();
        String patchStr = "";
        //注意最后一行
        for(int i = 0;i < list.size();++i){
            String str = list.get(i);
            if(str.isEmpty()){
                break;
            }
            String[] tmp = str.split(" ");
            String id = tmp[1];
            //testing
            if(IdSet.contains(id))
                continue;
            if(lastid.isEmpty() || lastid.equals(id)) {
                patchStr += cat2index.get(tmp[0]) + " ";
//                patch.add(cat2index.get(tmp[0]));//cat
                lastid = id;
            }else{
                File file = new File(rootPath + lastid + ".graph");
//                System.out.println(file.getName() + " " + file.exists());
                if (file.exists()){
                    bw.write(FileUtils.readFileToString(file) + patchStr + "\r\n");
//                    existList.add(lastid);
                }
                patchStr = cat2index.get(tmp[0]) + " ";
                lastid = id;
            }
        }
        File file = new File(rootPath + lastid + ".graph");
        if (file.exists()){
//            existList.add(lastid);
            bw.write(FileUtils.readFileToString(file) + patchStr + "\r\n");
        }
        patch.clear();

        System.out.println("completed!");
        bw.close();
        writer.close();
//        FileUtils.writeLines(new File(pathPatch+"existListForRouterGraphs"),existList);

        return 0;
    }

    private void heyuRequest() throws IOException {
        List<String> list = FileUtils.readLines(new File(pathPatch+"LSHTC/remap.txt"));
        HashMap<String,String> label2index = new HashMap<>();
        for(String labelTuple:list){
            int index = labelTuple.indexOf(" ");
            label2index.put(labelTuple.substring(1 + index),labelTuple.substring(0,index));
        }
        list = FileUtils.readLines(new File(pathPatch+"data_LSHTC_All/train_graphs"));
        List<String> list1 = new ArrayList<>() , list2 = new ArrayList<>();
        for(int i = 0;i < list.size();i++){
            if((i+1) % 5 == 0){
                String[] tmp = list.get(i).split(" ");
                String res = "";
                for(String label:tmp){
                    res += label2index.get(label) + " ";
                }
                list2.add(res);
            }else{
                list1.add(list.get(i));
            }
        }
        FileUtils.writeLines(new File(pathPatch+"data_LSHTC_All/pamer/train_graphs"),list1);
        FileUtils.writeLines(new File(pathPatch+"data_LSHTC_All/pamer/train_labels"),list2);

        list = FileUtils.readLines(new File(pathPatch+"data_LSHTC_All/test_graphs"));
        list1.clear();
        list2.clear();
        for(int i = 0;i < list.size();i++){
            if((i+1) % 5 == 0){
                String[] tmp = list.get(i).split("\\s+");
                String res = "";
                for(String label:tmp){
                    res += label2index.get(label) + " ";
                }
                list2.add(res);
            }else{
                list1.add(list.get(i));
            }
        }
        FileUtils.writeLines(new File(pathPatch+"data_LSHTC_All/pamer/test_graphs"),list1);
        FileUtils.writeLines(new File(pathPatch+"data_LSHTC_All/pamer/test_labels"),list2);
    }

    public void test() throws IOException {
        printWord2IndexFill();
        long starTime=System.currentTimeMillis();
        // creates a StanfordCoreNLP object, with POS tagging, lemmatization, NER, parsing, and coreference resolution
        Properties props = new Properties();
        props.setProperty("annotators", "tokenize, cleanxml, ssplit, pos, lemma, ner , parse");//, parse
        //ner: PER LOC ORG MISC
        //parse: agent advcl appos ccomp csubj csubjpass dobj
        StanfordCoreNLP pipeline = new StanfordCoreNLP(props);

        //Added
//        HashMap<String,Integer> word2Index = new HashMap<>();
        int wordCnt = 0;
        // init weight for node & edge
        float nodeWeightList[] = {2.4f,1.19f,2.6f};
        HashMap<String,Float> edgeWeightMap = new HashMap<>();
        initEdgeWeightMap(edgeWeightMap);
        HashMap<String,Integer> stopWordMap = new HashMap<>();

        //files for test
//        String rootFileName = pathPatch + "InputFiles";
//        File root = new File(rootFileName);
//        File[] files = root.listFiles();
        //stopword
        List<String> stopList = FileUtils.readLines(new File(pathPatch+"english.stop"));
        for(String stopword:stopList){
            stopWordMap.put(stopword,0);
        }

        //word2vec
//        FileWriter writer = null;
//        writer = new FileWriter("D:/DataSet/Song teacher/rcv1v2/ReutersCorpusVolume1/Content.txt");
//        BufferedWriter bw = new BufferedWriter(writer);

        //use
        int filesCnt = 0;
        List<String> fileList = null;
        fileList = FileUtils.readLines(new File(pathPatch+"heyu/rcv1v2-ids.dat"));//-23149
        String outPathPrefix = pathPatch+"RouterGraph/";//_23149
        int updateFlag = 0;
//        String lastUpdate = FileUtils.readFileToString(new File("D:/DataSet/Song teacher/rcv1v2/ReutersCorpusVolume1/lastUpdate"));

        int left = inner * range,right = left + range;
        if(inner == 24)
            right = fileList.size();

        for(int inn = left;inn < right;++inn){
            String prefix = fileList.get(inn);
            File outFile = new File(outPathPrefix + prefix + ".graph");
            if(outFile.exists())
                continue;
//            if(updateFlag == 0){
//                if(lastUpdate.equals(prefix)) {
//                    updateFlag = 1;
//                }
//                continue;
//            }
            File file = new File(pathPatch+"Data/ReutersCorpusVolume1_Original/CD1/"+prefix+"newsML.xml");
            if(!file.exists())
                file = new File(pathPatch+"Data/ReutersCorpusVolume1_Original/CD2/"+prefix+"newsML.xml");
            if(!file.exists())
                continue;
            // read some text in the text variable
            String text = processFile(file);
            text = text.replaceAll("\t"," ");
//            System.out.println(text);
            System.out.println(file.getName() + "\n" + text);
            if(text.isEmpty())
                continue;

            // create an empty Annotation just with the given text
            Annotation document = new Annotation(text);

            // run all Annotators on this text
            pipeline.annotate(document);
            List<CoreMap> sentences = document.get(SentencesAnnotation.class);
            HashMap<String, WordAttr> weightMap = new HashMap<>();
            int wordCntForOneFile = 0;

//            System.out.println("word\tpos\tlemma\tner");
//            writer = new FileWriter("D:/DataSet/Song teacher/rcv1v2/ReutersCorpusVolume1/Content.txt",true);
//            bw = new BufferedWriter(writer);
            for (CoreMap sentence : sentences) {
                // traversing the words in the current sentence
                // a CoreLabel is a CoreMap with additional token-specific methods
                for (CoreLabel token : sentence.get(TokensAnnotation.class)) {
                    // this is the text of the token
                    String word = token.get(TextAnnotation.class);
                    // this is the POS tag of the token
                    String pos = token.get(PartOfSpeechAnnotation.class);
                    // this is the NER label of the token
                    String ne = token.get(NamedEntityTagAnnotation.class);
                    String lemma = token.get(LemmaAnnotation.class);

//                    System.out.println(word + "\t" + pos + "\t" + lemma + "\t" + ne);

                    if(!printWord2Index.containsKey(lemma))//attention
                        continue;

                    int flag = -1;
                    //只添加动词 & 名词 & 杂项
//                    System.out.println(word + " " + ne);
                    if(ne.startsWith("MISC") || ne.equals("O")) {
                        if (pos.startsWith("NN")) {
                            flag = 2;
                            if (ne.startsWith("MISC"))
                                flag = 0;
                        } else if (pos.startsWith("VB")) flag = 1;
                    }

                    if (flag > -1) {
                        if(stopWordMap.containsKey(lemma))
                            continue;
//                        if (word2Index.containsKey(lemma)) {
//                        } else {
//                            word2Index.put(lemma, wordCnt);
//                            wordCnt++;
//                        }
//                        bw.write(lemma + " ");
//                        System.out.println(lemma);

                        float w = nodeWeightList[flag];
                        if (weightMap.containsKey(lemma)) {
                            WordAttr wordAttr = weightMap.get(lemma);
                            wordAttr.time ++;
                        } else {
                            weightMap.put(lemma, new WordAttr(wordCntForOneFile, w ));
                            wordCntForOneFile++;
                        }
                    }
                }
//                bw.write("\r\n");
            }
//            if(filesCnt % 1000 == 0) {
//                System.out.println(wordCnt + " words & " + filesCnt + " files");
//                long endTime=System.currentTimeMillis();
//                System.out.println( (endTime-starTime) / 1000.0 / 60 + " min used!");
//            }
//            FileUtils.writeStringToFile(new File("D:/DataSet/Song teacher/rcv1v2/ReutersCorpusVolume1/lastUpdate"),prefix);
//            bw.close();
//            writer.close();
//            if(filesCnt++ > -1)
//                continue;

            float linkWeight[][] = new float[wordCntForOneFile][wordCntForOneFile];
            int linkTime[][] = new int[wordCntForOneFile][wordCntForOneFile];
            for (CoreMap sentence : sentences) {
                // 获取dependency graph
                SemanticGraph dependencies = sentence.get(SemanticGraphCoreAnnotations.CollapsedCCProcessedDependenciesAnnotation.class);

                List<SemanticGraphEdge> edgeList = new ArrayList();
                Iterator var2 = dependencies.edgeIterable().iterator();

                while(var2.hasNext()) {
                    SemanticGraphEdge edge = (SemanticGraphEdge)var2.next();
                    edgeList.add(edge);
                }

                for (SemanticGraphEdge semanticGraphEdge : edgeList) {
                    String dep = semanticGraphEdge.getTarget().lemma();
                    String gov = semanticGraphEdge.getSource().lemma();
                    if (!weightMap.containsKey(dep) || !weightMap.containsKey(gov))
                        continue;
                    int govi = weightMap.get(gov).wordPos, depi = weightMap.get(dep).wordPos;
                    String rel = semanticGraphEdge.getRelation().getShortName();
                    float w = 1;
//                    System.out.println(gov + "->" + dep + "(" + rel + ")");
                    if (edgeWeightMap.containsKey(rel)) {
                        w = edgeWeightMap.get(rel);
                    } else {
//                        System.out.println(gov + "->" + dep + "(" + rel + ")");
                    }
                    linkWeight[govi][depi] += w;
                    linkTime[govi][depi]++;
                    if(rel.equals("compound")){
                        linkWeight[depi][govi] += w;
                        linkTime[depi][govi]++;
                    }
                }
            }

            //need to notes
//            printWord2Index = word2Index;

            int[] list1 = new int[wordCntForOneFile];
            float[] list2 = new float[wordCntForOneFile];
            String[] list3 = new String[wordCntForOneFile];
            Set<Map.Entry<String, WordAttr>> entrySet = weightMap.entrySet();
            for (Map.Entry<String, WordAttr> entry:entrySet) {
                String word = entry.getKey();
                WordAttr wordAttr = entry.getValue();
                if(printWord2Index.containsKey(word)){
                    list1[wordAttr.wordPos] = printWord2Index.get( word );
                }
                list2[wordAttr.wordPos] = wordAttr.weight * wordAttr.time * (float) Math.pow(unit,wordAttr.time - 1);
                list3[wordAttr.wordPos] = word;
            }

            String outStr = "" + wordCntForOneFile + "\r\n";
//            System.out.println(wordCntForOneFile);
            for (int i = 0;i < list1.length;++i) {
//                System.out.print(list1[i] + " ");
                outStr += list2[i] + " ";
            }
//            System.out.println();
            outStr += "\r\n";
//            for (int i = 0;i < list3.length;++i) {
////                System.out.print(list1[i] + " ");
//                outStr += list3[i] + " ";
//            }
//            outStr += "\r\n";
            for (int i = 0;i < list2.length;++i) {
//                System.out.print(list2[i] + " ");
                outStr += list1[i] + " ";
//                outStr += list3[i] + " " + list2[i] + " ";
            }
//            System.out.println();
            outStr += "\r\n";
            for(int i = 0;i < wordCntForOneFile;++i)
                for(int j = 0;j < wordCntForOneFile;++j){
                    linkWeight[i][j] = linkWeight[i][j] * (float)Math.pow(unit,linkTime[i][j]);
                    if(linkWeight[i][j] > 0){
//                        System.out.print(i + " " + j + " " + link[i][j] + " ");
                        outStr += i + " " + j + " " + linkWeight[i][j] + " ";
                    }
                }
//            System.out.println();
            outStr += "\r\n";
//            for(int i = 0;i < wordCntForOneFile;++i) {
//                for (int j = 0; j < wordCntForOneFile; ++j) {
//                    outStr += link[i][j] + " ";
//                }
//                outStr += "\r\n";
//            }
//            outStr += "\r\n";

            try {
                FileUtils.writeStringToFile(outFile , outStr);
            } catch (IOException e) {
                e.printStackTrace();
            }

            filesCnt++;
            if(filesCnt % 500 == 0) {
                System.out.println("process " + inner + ": " + filesCnt + " files completed!");
                long endTime=System.currentTimeMillis();
                System.out.println( (endTime-starTime) / 1000.0 / 60 + " min used!");
            }
//            if(filesCnt > 0){
//                break;
//            }
        }

        long endTime=System.currentTimeMillis();
        System.out.println( (endTime-starTime) / 1000.0 / 60 + " min used!");
    }

    public void Kim() throws IOException {
        final float coeffi = 1.48f;
        Properties props = new Properties();
        props.setProperty("annotators", "tokenize, cleanxml, ssplit, pos, lemma, ner , parse");
        StanfordCoreNLP pipeline = new StanfordCoreNLP(props);
        HashMap<String,Integer> word2Index = new HashMap<>();
        HashMap<String,Integer> label2Index1 = new HashMap<>(),label2Index2 = new HashMap<>();
        int wordCnt = 0,labelCnt1 = 0,labelCnt2 = 0;
        // init weight for node & edge
        float nodeWeightList[] = {1f,1.52f,1.56f,1.49f,1.45f};
        HashMap<String,Float> edgeWeightMap = new HashMap<>();
        initEdgeWeightMap(edgeWeightMap);

        List<String> list = FileUtils.readLines(new File("InputFiles/TREC/All.txt"));
        List<String> outputList = new ArrayList<>();
        List<String> labelList1 = new ArrayList<>(),labelList2 = new ArrayList<>();

        int cnt = 0;
        for(String str:list){

            int sp = str.indexOf(" ");
            String label = str.substring(0,sp);
            sp = label.indexOf(":");

            String text = str.substring(sp+1);
            text = text.replaceAll("-"," ");
//            text = "What is e coli ?";
            int wordCntForOneFile = 0;
            Annotation document = new Annotation(text);
            pipeline.annotate(document);
            List<CoreMap> sentences = document.get(SentencesAnnotation.class);
            HashMap<String, WordAttr> weightMap = new HashMap<>();

            int nameFlag = 0;
            for (CoreMap sentence : sentences) {
                for (CoreLabel token : sentence.get(TokensAnnotation.class)) {
                    String word = token.get(TextAnnotation.class);
                    String pos = token.get(PartOfSpeechAnnotation.class);
                    String ne = token.get(NamedEntityTagAnnotation.class);
                    String lemma = token.get(LemmaAnnotation.class).toLowerCase();

//                    System.out.println(word + "\t" + pos + "\t" + lemma + "\t" + ne);

                    //去空格和未出现词随机向量的选择
                    if (!vec.hasWord(lemma))
                        continue;
//                    if(pos.charAt(0) < 'A' || pos.charAt(0) > 'Z')//去标点
//                        continue;

                    int flag = 0;
                    if (pos.startsWith("NN")) {
                        flag = 4;
                    } else if (pos.startsWith("VB")) flag = 3;
                    else if (pos.startsWith("W")) flag = 2;

                    if (nameFlag == 0) {
                        if (pos.startsWith("VB")) flag = 1;
                        nameFlag = 1;
                    }

                    if (!word2Index.containsKey(lemma)) {
                        word2Index.put(lemma, wordCnt);
                        wordCnt++;
                    }

                    if (weightMap.containsKey(lemma)) {
                        WordAttr wordAttr = weightMap.get(lemma);
                        wordAttr.weight += nodeWeightList[flag];
                    } else {
                        WordAttr wordAttr = new WordAttr(wordCntForOneFile, nodeWeightList[flag]);
                        if (flag == 1 || flag == 2) {
                            wordAttr.time = 2;// time means important or not here
                        }
                        weightMap.put(lemma, wordAttr);
                        wordCntForOneFile++;
                    }
                }
            }
            float linkWeight[][] = new float[wordCntForOneFile][wordCntForOneFile];
            for (CoreMap sentence : sentences) {
                SemanticGraph dependencies = sentence.get(SemanticGraphCoreAnnotations.CollapsedCCProcessedDependenciesAnnotation.class);

                List<SemanticGraphEdge> edgeList = new ArrayList();
                Iterator var2 = dependencies.edgeIterable().iterator();

                while(var2.hasNext()) {
                    SemanticGraphEdge edge = (SemanticGraphEdge)var2.next();
                    edgeList.add(edge);
                }

                for (SemanticGraphEdge semanticGraphEdge : edgeList) {
                    String dep = semanticGraphEdge.getTarget().lemma().toLowerCase();
                    String gov = semanticGraphEdge.getSource().lemma().toLowerCase();
                    if (!weightMap.containsKey(dep) || !weightMap.containsKey(gov))
                        continue;
                    int govi = weightMap.get(gov).wordPos, depi = weightMap.get(dep).wordPos;
                    String rel = semanticGraphEdge.getRelation().getShortName();
//                    System.out.println(gov + "->" + dep + "(" + rel + ")");

                    linkWeight[govi][depi] ++;
                    if(rel.equals("compound")){
                        linkWeight[depi][govi] ++;
                    }
                }
            }

            //need to notes
            printWord2Index = word2Index;

            int[] list1 = new int[wordCntForOneFile];
            float[] list2 = new float[wordCntForOneFile];
            int[] list3 = new int[wordCntForOneFile];
            Set<Map.Entry<String, WordAttr>> entrySet = weightMap.entrySet();
            for (Map.Entry<String, WordAttr> entry:entrySet) {
                String word = entry.getKey();
                WordAttr wordAttr = entry.getValue();
                if(!printWord2Index.containsKey(word)){
                    printWord2Index.put(word,printCnt);
                    list1[wordAttr.wordPos] = printCnt;
                    printCnt++;
                }else{
                    list1[wordAttr.wordPos] = printWord2Index.get( word );
                }
                list2[wordAttr.wordPos] = wordAttr.weight;
                list3[wordAttr.wordPos] = wordAttr.time;
            }

            outputList.add(wordCntForOneFile+"");
            String outStr = "";
//            System.out.println(wordCntForOneFile);
            for (int i = 0;i < list1.length;++i) {
//                System.out.print(list1[i] + " ");
                outStr += list2[i] + " ";
            }
//            System.out.println();
            outputList.add(outStr); outStr = "";
            for (int i = 0;i < list2.length;++i) {
//                System.out.print(list2[i] + " ");
                outStr += list1[i] + " ";
//                outStr += list3[i] + " " + list2[i] + " ";
            }
//            System.out.println();
            outputList.add(outStr); outStr = "";
            for(int i = 0;i < wordCntForOneFile;++i)
                for(int j = 0;j < wordCntForOneFile;++j){
                    if((list3[i] + list3[j]) > 2)
                        linkWeight[i][j] *= coeffi;
                    if(linkWeight[i][j] > 0){
                        outStr += i + " " + j + " " + linkWeight[i][j] + " ";
                    }
                }
//            System.out.println();
            outputList.add(outStr); outStr = "";

            String label1 = label.substring(0,sp),label2 = label.substring(sp+1);
            if(!label2Index1.containsKey(label1)){
                label2Index1.put(label1,labelCnt1);
                labelList1.add(labelCnt1 + " " + label1);
                outStr += labelCnt1;
                labelCnt1++;
            }else
                outStr += label2Index1.get(label1);
            outputList.add(outStr); outStr = "";

            if(!label2Index2.containsKey(label2)){
                label2Index2.put(label2,labelCnt2);
                labelList2.add(labelCnt2 + " " + label2);
                outStr += labelCnt2;
                labelCnt2++;
            }else
                outStr += label2Index2.get(label2);
            outputList.add(outStr);

//            if(cnt++ >= 0)
//                break;
        }


        FileUtils.writeLines(new File("InputFiles/TREC/labels1"),labelList1);
        FileUtils.writeLines(new File("InputFiles/TREC/labels2"),labelList2);
        FileUtils.writeLines(new File("InputFiles/TREC/graphs"),outputList);
        int graphCnt = outputList.size()/6;
        outputList.clear();
        outputList.add(graphCnt+"");
        outputList.add(labelCnt1+"");outputList.add(labelCnt2+"");
        outputList.add(vec.getSize()+"");
        FileUtils.writeLines(new File("InputFiles/TREC/option"),outputList);
        printVectors("InputFiles/TREC/index2vec");
    }

    public void LSHTC() throws IOException {
        printWord2IndexFill();
        long starTime=System.currentTimeMillis();
        Properties props = new Properties();
        props.setProperty("annotators", "tokenize, cleanxml, ssplit, pos, lemma, ner , parse");//, parse

        StanfordCoreNLP pipeline = new StanfordCoreNLP(props);

        //Added
        int wordCnt = 0;
        // init weight for node & edge
        float nodeWeightList[] = {2.4f,1.19f,2.6f};
        HashMap<String,Float> edgeWeightMap = new HashMap<>();
        initEdgeWeightMap(edgeWeightMap);
        HashMap<String,Integer> stopWordMap = new HashMap<>();

        //files for test
//        String rootFileName = pathPatch + "InputFiles";
//        File root = new File(rootFileName);
//        File[] files = root.listFiles();
        //stopword
        List<String> stopList = FileUtils.readLines(new File(pathPatch+"english.stop"));
        for(String stopword:stopList){
            stopWordMap.put(stopword,0);
        }

        //word2vec
//        FileWriter writer = null;
//        writer = new FileWriter(pathPatch+"LSHTC/ContentAll.txt");
//        BufferedWriter bw = new BufferedWriter(writer);

        //use
        int filesCnt = 0;
        List<String> fileList = null;
        fileList = FileUtils.readLines(new File(pathPatch+"LSHTC/wikipediaMediumOriginalLSHTCv3-train.txt"));
        fileList.addAll(FileUtils.readLines(new File(pathPatch+"LSHTC/wikipediaMediumOriginalLSHTCv3-test.txt")));
        String outPathPrefix = pathPatch+"LSHTC/GraphAll/";

        int left = inner * range + 1,right = left + range;
        if(inner == 24)
            right = (fileList.size() / 5) + 1;

        //word2vec
//        left = 1;
//        right = fileList.size() / 5 + 1;

        for(int inn = left;inn < right;++inn){
//            String prefixHas = fileList.get(inn*5-4);
//            File outFile = new File(outPathPrefix + prefixHas.substring(prefixHas.indexOf('>')+1,prefixHas.lastIndexOf('<')) + ".graph");
            File outFile = new File(outPathPrefix + inn + ".graph");
            if(outFile.exists())
                continue;
            String label = fileList.get(inn*5-3);
            label = label.substring(label.indexOf('>')+1,label.lastIndexOf('<'));

            String text = fileList.get(inn * 5 - 2);
            text = text.replaceAll("\t"," ");
//            int cntEnter = 1;
//            for(int ff = 0;ff < text.length();++ff){
//                if(text.charAt(ff) == ' '){
//                    cntEnter++;
//                }
//                if(cntEnter % 20 == 0)
//                    text = text.substring(0,ff)+'.'+text.substring(ff);
//            }
//            System.out.println(text);
//            System.out.println("process " + inner + ": " + "\n" + text);

            // create an empty Annotation just with the given text
            Annotation document = new Annotation(text);

            // run all Annotators on this text
            pipeline.annotate(document);
            List<CoreMap> sentences = document.get(SentencesAnnotation.class);
            HashMap<String, WordAttr> weightMap = new HashMap<>();
            int wordCntForOneFile = 0;

//            System.out.println("word\tpos\tlemma\tner");
//            writer = new FileWriter("D:/DataSet/Song teacher/rcv1v2/ReutersCorpusVolume1/Content.txt",true);
//            bw = new BufferedWriter(writer);
            for (CoreMap sentence : sentences) {
                // traversing the words in the current sentence
                // a CoreLabel is a CoreMap with additional token-specific methods
//                String writeStr = "";
                for (CoreLabel token : sentence.get(TokensAnnotation.class)) {
                    // this is the text of the token
                    String word = token.get(TextAnnotation.class);
                    // this is the POS tag of the token
                    String pos = token.get(PartOfSpeechAnnotation.class);
                    // this is the NER label of the token
                    String ne = token.get(NamedEntityTagAnnotation.class);
                    String lemma = token.get(LemmaAnnotation.class).toLowerCase();

//                    System.out.println(word + "\t" + pos + "\t" + lemma + "\t" + ne);

                    if(!printWord2Index.containsKey(lemma))//attention
                        continue;

                    int flag = -1;
                    //只添加动词 & 名词 & 杂项
//                    System.out.println(word + " " + ne);
                    if(ne.startsWith("MISC") || ne.equals("O")) {
                        if (pos.startsWith("NN")) {
                            flag = 2;
                            if (ne.startsWith("MISC"))
                                flag = 0;
                        } else if (pos.startsWith("VB")) flag = 1;
                    }

                    if(stopWordMap.containsKey(lemma))
                        continue;
                    if(pos.charAt(0) < 'A' || pos.charAt(0) > 'Z' || pos.startsWith("SYM") || pos.startsWith("LS"))
                        continue;
//                    writeStr += lemma + " ";
//                    System.out.println(lemma);

                    float w = 1f;
                    if(flag > -1)
                        w = nodeWeightList[flag];
                    if (weightMap.containsKey(lemma)) {
                        WordAttr wordAttr = weightMap.get(lemma);
                        wordAttr.time ++;
                    } else {
                        weightMap.put(lemma, new WordAttr(wordCntForOneFile, w ));
                        wordCntForOneFile++;
                    }
                }
//                if(!writeStr.isEmpty())
//                    bw.write(writeStr + "\r\n");
            }
//            if(filesCnt % 1000 == 0) {
//                System.out.println(wordCnt + " words & " + filesCnt + " files");
//                long endTime=System.currentTimeMillis();
//                System.out.println( (endTime-starTime) / 1000.0 / 60 + " min used!");
//            }
//            bw.close();
//            writer.close();
//            if(filesCnt++ > -1)
//                continue;

            float linkWeight[][] = new float[wordCntForOneFile][wordCntForOneFile];
            int linkTime[][] = new int[wordCntForOneFile][wordCntForOneFile];
            for (CoreMap sentence : sentences) {
                // 获取dependency graph
                SemanticGraph dependencies = sentence.get(SemanticGraphCoreAnnotations.CollapsedCCProcessedDependenciesAnnotation.class);

                List<SemanticGraphEdge> edgeList = new ArrayList();
                Iterator var2 = dependencies.edgeIterable().iterator();

                while(var2.hasNext()) {
                    SemanticGraphEdge edge = (SemanticGraphEdge)var2.next();
                    edgeList.add(edge);
                }

                for (SemanticGraphEdge semanticGraphEdge : edgeList) {
                    String dep = semanticGraphEdge.getTarget().lemma();
                    String gov = semanticGraphEdge.getSource().lemma();
                    if(dep != null)dep = dep.toLowerCase();
                    if(gov != null)gov = gov.toLowerCase();
                    if (!weightMap.containsKey(dep) || !weightMap.containsKey(gov))
                        continue;
                    int govi = weightMap.get(gov).wordPos, depi = weightMap.get(dep).wordPos;
                    String rel = semanticGraphEdge.getRelation().getShortName();
                    float w = 1f;
//                    System.out.println(gov + "->" + dep + "(" + rel + ")");
                    if (edgeWeightMap.containsKey(rel)) {
                        w = edgeWeightMap.get(rel);
                    } else {
//                        System.out.println(gov + "->" + dep + "(" + rel + ")");
                    }
                    linkWeight[govi][depi] += w;
                    linkTime[govi][depi]++;
                    if(rel.equals("compound")){
                        linkWeight[depi][govi] += w;
                        linkTime[depi][govi]++;
                    }
                }
            }

            //need to notes
//            printWord2Index = word2Index;

            int[] list1 = new int[wordCntForOneFile];
            float[] list2 = new float[wordCntForOneFile];
            String[] list3 = new String[wordCntForOneFile];
            Set<Map.Entry<String, WordAttr>> entrySet = weightMap.entrySet();
            for (Map.Entry<String, WordAttr> entry:entrySet) {
                String word = entry.getKey();
                WordAttr wordAttr = entry.getValue();
                if(printWord2Index.containsKey(word)){
                    list1[wordAttr.wordPos] = printWord2Index.get( word );
                }
                list2[wordAttr.wordPos] = wordAttr.weight * wordAttr.time * (float) Math.pow(unit,wordAttr.time - 1);
                list3[wordAttr.wordPos] = word;
            }

            String outStr = "" + wordCntForOneFile + "\r\n";
//            System.out.println(wordCntForOneFile);
            for (int i = 0;i < list1.length;++i) {
//                System.out.print(list1[i] + " ");
                outStr += list2[i] + " ";
            }
//            System.out.println();
            outStr += "\r\n";
//            for (int i = 0;i < list3.length;++i) {
////                System.out.print(list1[i] + " ");
//                outStr += list3[i] + " ";
//            }
//            outStr += "\r\n";
            for (int i = 0;i < list2.length;++i) {
//                System.out.print(list2[i] + " ");
                outStr += list1[i] + " ";
//                outStr += list3[i] + " " + list2[i] + " ";
            }
//            System.out.println();
            outStr += "\r\n";
            for(int i = 0;i < wordCntForOneFile;++i)
                for(int j = 0;j < wordCntForOneFile;++j){
                    linkWeight[i][j] = linkWeight[i][j] * (float)Math.pow(unit,linkTime[i][j]);
                    if(linkWeight[i][j] > 0){
//                        System.out.print(i + " " + j + " " + link[i][j] + " ");
                        outStr += i + " " + j + " " + linkWeight[i][j] + " ";
                    }
                }
//            System.out.println();
            outStr += "\r\n" + label + "\r\n";
//            for(int i = 0;i < wordCntForOneFile;++i) {
//                for (int j = 0; j < wordCntForOneFile; ++j) {
//                    outStr += link[i][j] + " ";
//                }
//                outStr += "\r\n";
//            }
//            outStr += "\r\n";

            try {
                FileUtils.writeStringToFile(outFile , outStr);
            } catch (IOException e) {
                e.printStackTrace();
            }
//            System.exit(-1);
            System.out.println(filesCnt);
            filesCnt++;
            if(filesCnt % 1000 == 0) {
                System.out.println("process " + +inner + ": " + filesCnt + " files completed!");
                long endTime=System.currentTimeMillis();
                System.out.println( (endTime-starTime) / 1000.0 / 60 + " min used!");
            }
//            if(filesCnt > 0){
//                break;
//            }
        }

//        bw.close();
//        writer.close();

        long endTime=System.currentTimeMillis();
        System.out.println("process " + inner + " finally use" + (endTime-starTime) / 1000.0 / 60 + "minutes!");
    }

    public void NYTimesLabel() throws IOException {
        String rootFileName = pathPatch + "NYT_annotated_corpus/data_all";
        File root = new File(rootFileName);
        File[] files = root.listFiles();

        String outPathPrefix = pathPatch+"NYT/Label_new/";
        String outPathForLabelMap = pathPatch+"NYT/labelMap_new.txt" , outPathForLabelInherit = pathPatch+"NYT/labelInherit_new.txt";
        String outPathForleaveRemap = pathPatch+"NYT/remap_new.txt" , outPathForlabelTimes = pathPatch+"NYT/labelTimes.txt";
        //labelInherit.txt记录 标签 父标签 所处级别
        HashMap<Integer,List<Integer>> fatherMap = new HashMap<>();
        HashMap<String,Integer> labelMap = new HashMap<>();
        HashMap<Integer,Integer> leaveRemap = new HashMap<>();
        HashMap<Integer,List<Integer>> sonMap = new HashMap<>();

        List<Integer> times = new ArrayList<>();
        int needTo = 2318;
        while (needTo > 0){times.add(0);needTo --;}
        int labelCnt = 0 , leaveCnt = 0;
        int left = 0,right = files.length;

        int maxLevel = 0 , debug = 0;
        String result = "";
        Set<Integer> rmDuplicate = new HashSet<>();
        for(int inn = left;inn < right;++inn) {
            File outFile = new File(outPathPrefix + inn + ".label");
//            if (outFile.exists())
//                continue;

            String label = "";

            List<String> lines = FileUtils.readLines(files[inn]);
//            System.out.println(files[inn].getName());
            List<String> labelList = new ArrayList<>();
            String tmp;
            result = "";
            int flag = 0;
            for(int i = 0;i < lines.size();++i){
                tmp = lines.get(i).trim();
                if(flag == 1 && tmp.contains("taxonomic_classifier")) {
                    int x = tmp.indexOf('>') + 1, y = tmp.lastIndexOf('<');
                    if (x < y) {
                        tmp = tmp.substring(x, y);
//                        result += tmp + "\r\n";
                        labelList.add(tmp);
                    }
                }
                if(tmp.startsWith("<classifier"))
                    flag = 1;
                else if(tmp.equals("</identified-content>"))
                    break;
            }
            if(debug == 1){
                for(String str:labelList) {
                    String[] strs = str.split("/");
                    maxLevel = Math.max(maxLevel , strs.length);
                }
                if(inn % 10000 == 0) {
                    System.out.println(inn + " files completed!");
                }
                continue;
            }
//            System.out.println(result);
            result = "";
            HashMap<String,Integer> tmpMap = new HashMap<>();
            for(String str:labelList){
                String[] strs = str.split("/");
                int[] strsNum = new int[strs.length];
                for(int i = 0;i < strs.length;++i){
                    String lab = strs[i];
                    if(labelMap.containsKey(lab)){
                        int labNum = labelMap.get(lab);
                        strsNum[i] = labNum;
                    }else{
                        labelMap.put(lab,labelCnt);
                        strsNum[i] = labelCnt;
                        labelCnt++;
                    }
                    if(!tmpMap.containsKey(lab) && labelMap.containsKey(lab)){
                        tmpMap.put(lab,0);
                        int ind = labelMap.get(lab);
                        times.set( ind , times.get(ind) + 1);
                    }
                }
//                rmDuplicate.add(strsNum[strs.length - 1]);
                if(!leaveRemap.containsKey(strsNum[strs.length - 1])) {
                    leaveRemap.put(strsNum[strs.length - 1], leaveCnt);
                    leaveCnt++;
                }
                if(debug == 0)continue;
                for(int i = 0;i < strs.length;++i){
//                    rmDuplicate.add(strsNum[i]);

//                    result += strsNum[i] + " ";
                    if(i > 0){
                        if(!fatherMap.containsKey(strsNum[i])) {
                            List<Integer> list = new ArrayList<>();
                            list.add(strsNum[i - 1]);
                            fatherMap.put(strsNum[i], list);
                        }else{
                            List<Integer> list = fatherMap.get(strsNum[i]);
                            if(!list.contains(strsNum[i - 1])) {
                                list.add(strsNum[i - 1]);
                                fatherMap.put(strsNum[i], list);
                            }
                        }
                    }
                    if(i < strs.length - 1){
                        if(!sonMap.containsKey(strsNum[i])) {
                            List<Integer> list = new ArrayList<>();
                            list.add(strsNum[i + 1]);
                            sonMap.put(strsNum[i], list);
                        }else{
                            List<Integer> list = sonMap.get(strsNum[i]);
                            if(!list.contains(strsNum[i + 1])) {
                                list.add(strsNum[i + 1]);
                                sonMap.put(strsNum[i], list);
                            }
                        }
                    }
                }
            }
//            Iterator iter = rmDuplicate.iterator();
//            while(iter.hasNext())
//                result += iter.next() + " ";
//
//            FileUtils.writeStringToFile(new File(outPathPrefix + inn + ".label"), result);
            //debug
            rmDuplicate.clear();

            if(inn % 1000 == 0) {
                System.out.println(inn + " files completed!");
            }
        }
        if(debug == 1){
            System.out.println("maxLevel = "+maxLevel);
            return;
        }

        FileUtils.writeLines(new File(outPathForlabelTimes),times);
        if(debug == 0)return;

        List<String> output = new ArrayList<>();
        Set<Map.Entry<String, Integer>> entryseSet = labelMap.entrySet();
        for (Map.Entry<String, Integer> entry:entryseSet) {
            output.add(entry.getValue() + " " + entry.getKey());
        }
        FileUtils.writeLines(new File(outPathForLabelMap),output);
        output.clear();

        Set<Map.Entry<Integer, Integer>> entrySet = leaveRemap.entrySet();
        for (Map.Entry<Integer, Integer> entry:entrySet) {
            output.add(entry.getKey() + " " + entry.getValue());
        }
        FileUtils.writeLines(new File(outPathForleaveRemap),output);
        output.clear();


        for(int i = 0;i < labelCnt;++i){
            output.add(i + "");
            List<Integer> list = fatherMap.get(i);
            result = "";
            if(list != null) {
                for (Integer integer : list)
                    result += integer + " ";
            }
            output.add(result);

            list = sonMap.get(i);
            result = "";
            if(list != null) {
                for (Integer integer : list)
                    result += integer + " ";
            }
            output.add(result);
        }
        FileUtils.writeLines(new File(outPathForLabelInherit),output);

    }

    public void NYTimes() throws IOException {
        printWord2IndexFill();
        long starTime=System.currentTimeMillis();
        Properties props = new Properties();
        props.setProperty("annotators", "tokenize, ssplit, pos, lemma, ner , parse");//, parse

        StanfordCoreNLP pipeline = new StanfordCoreNLP(props);

        //Added
        int wordCnt = 0;
        // init weight for node & edge
        float nodeWeightList[] = {2.4f,1.19f,2.6f};
        HashMap<String,Float> edgeWeightMap = new HashMap<>();
        initEdgeWeightMap(edgeWeightMap);
        HashMap<String,Integer> stopWordMap = new HashMap<>();

        //files for test
        String rootFileName = pathPatch + "NYT_annotated_corpus/data_all";
        File root = new File(rootFileName);
        File[] files = root.listFiles();
        //stopword
        List<String> stopList = FileUtils.readLines(new File(pathPatch+"english.stop"));
        for(String stopword:stopList){
            stopWordMap.put(stopword,0);
        }

        String labelPathPrefix = pathPatch+"NYT/Label/";

        //word2vec
//        FileWriter writer = null;
//        writer = new FileWriter(pathPatch+"NYT/Content.txt",true);
//        BufferedWriter bw = new BufferedWriter(writer);

        //use
        int filesCnt = 0;
//        List<String> fileList = null;
//        fileList = FileUtils.readLines(new File(pathPatch+"LSHTC/wikipediaMediumOriginalLSHTCv3-train.txt"));
//        fileList.addAll(FileUtils.readLines(new File(pathPatch+"LSHTC/wikipediaMediumOriginalLSHTCv3-test.txt")));
        String outPathPrefix = pathPatch+"NYT/Graph/";

        int left = 1391700 + inner * range ,right = left + range;
        if(inner == threadEnd)
            right = files.length;

        //word2vec
        //left = 980000;//980000~990000 & 1174789
        //1174789太长无法处理
        //right = 990000;

        for(int inn = left;inn < right;++inn){
//            String prefixHas = fileList.get(inn*5-4);
//            File outFile = new File(outPathPrefix + prefixHas.substring(prefixHas.indexOf('>')+1,prefixHas.lastIndexOf('<')) + ".graph");
            File outFile = new File(outPathPrefix + inn + ".graph");
            if(outFile.exists())
                continue;

            String text = processNYT(files[inn]);
//            System.out.println(inn+"\r\n"+text);
//            text = text.replaceAll("\t"," ");

            // create an empty Annotation just with the given text
            Annotation document = new Annotation(text);

            // run all Annotators on this text
            pipeline.annotate(document);
            List<CoreMap> sentences = document.get(SentencesAnnotation.class);
            HashMap<String, WordAttr> weightMap = new HashMap<>();
            int wordCntForOneFile = 0;

//            System.out.println("word\tpos\tlemma\tner");
//            writer = new FileWriter(pathPatch+"NYT/Content.txt",true);
//            bw = new BufferedWriter(writer);
            for (CoreMap sentence : sentences) {
                // traversing the words in the current sentence
                // a CoreLabel is a CoreMap with additional token-specific methods

                //word2vec
//                String writeStr = "";
                for (CoreLabel token : sentence.get(TokensAnnotation.class)) {
                    // this is the text of the token
                    String word = token.get(TextAnnotation.class);
                    // this is the POS tag of the token
                    String pos = token.get(PartOfSpeechAnnotation.class);
                    // this is the NER label of the token
                    String ne = token.get(NamedEntityTagAnnotation.class);
                    String lemma = token.get(LemmaAnnotation.class).toLowerCase();

//                    System.out.println(word + "\t" + pos + "\t" + lemma + "\t" + ne);

                    //attention
                    if(!printWord2Index.containsKey(lemma))
                        continue;

                    int flag = -1;
                    //只添加动词 & 名词 & 杂项
//                    System.out.println(word + " " + ne);
                    if(ne.startsWith("MISC") || ne.equals("O")) {
                        if (pos.startsWith("NN")) {
                            flag = 2;
                            if (ne.startsWith("MISC"))
                                flag = 0;
                        } else if (pos.startsWith("VB")) flag = 1;
                    }

//                    if(stopWordMap.containsKey(lemma))
//                        continue;

                    //word2vec
//                    if(pos.charAt(0) < 'A' || pos.charAt(0) > 'Z' || pos.startsWith("SYM") || pos.startsWith("LS")
//                            || pos.startsWith("POS") || pos.startsWith("CD"))
//                        continue;
//                    writeStr += lemma + " ";

//                    System.out.println(lemma);

                    float w = 1f;
                    if(flag > -1)
                        w = nodeWeightList[flag];
                    if (weightMap.containsKey(lemma)) {
                        WordAttr wordAttr = weightMap.get(lemma);
                        wordAttr.time ++;
                    } else {
                        weightMap.put(lemma, new WordAttr(wordCntForOneFile, w ));
                        wordCntForOneFile++;
                    }
                }
                //word2vec
//                if(!writeStr.isEmpty())
//                    bw.write(writeStr + "\r\n");
            }
            //word2vec
//            if(filesCnt % 500 == 0) {
//                System.out.println(wordCnt + " words & " + filesCnt + " files");
//                long endTime=System.currentTimeMillis();
//                System.out.println( (endTime-starTime) / 1000.0 / 60 + " min used!");
//            }
//            bw.close();
//            writer.close();
//            if(filesCnt++ > -1)
//                continue;

            float linkWeight[][] = new float[wordCntForOneFile][wordCntForOneFile];
            int linkTime[][] = new int[wordCntForOneFile][wordCntForOneFile];
            for (CoreMap sentence : sentences) {
                // 获取dependency graph
                SemanticGraph dependencies = sentence.get(SemanticGraphCoreAnnotations.CollapsedCCProcessedDependenciesAnnotation.class);

                List<SemanticGraphEdge> edgeList = new ArrayList();
                Iterator var2 = dependencies.edgeIterable().iterator();

                while(var2.hasNext()) {
                    SemanticGraphEdge edge = (SemanticGraphEdge)var2.next();
                    edgeList.add(edge);
                }

                for (SemanticGraphEdge semanticGraphEdge : edgeList) {
                    String dep = semanticGraphEdge.getTarget().lemma();
                    String gov = semanticGraphEdge.getSource().lemma();
                    if(dep != null)dep = dep.toLowerCase();
                    if(gov != null)gov = gov.toLowerCase();
                    if (!weightMap.containsKey(dep) || !weightMap.containsKey(gov))
                        continue;
                    int govi = weightMap.get(gov).wordPos, depi = weightMap.get(dep).wordPos;
                    String rel = semanticGraphEdge.getRelation().getShortName();
                    float w = 1f;
//                    System.out.println(gov + "->" + dep + "(" + rel + ")");
                    if (edgeWeightMap.containsKey(rel)) {
                        w = edgeWeightMap.get(rel);
                    } else {
//                        System.out.println(gov + "->" + dep + "(" + rel + ")");
                    }
                    linkWeight[govi][depi] += w;
                    linkTime[govi][depi]++;
                    if(rel.equals("compound")){
                        linkWeight[depi][govi] += w;
                        linkTime[depi][govi]++;
                    }
                }
            }

            //need to notes
//            printWord2Index = word2Index;

            int[] list1 = new int[wordCntForOneFile];
            float[] list2 = new float[wordCntForOneFile];
            String[] list3 = new String[wordCntForOneFile];
            Set<Map.Entry<String, WordAttr>> entrySet = weightMap.entrySet();
            for (Map.Entry<String, WordAttr> entry:entrySet) {
                String word = entry.getKey();
                WordAttr wordAttr = entry.getValue();
                if(printWord2Index.containsKey(word)){
                    list1[wordAttr.wordPos] = printWord2Index.get( word );
                }
                list2[wordAttr.wordPos] = wordAttr.weight * wordAttr.time * (float) Math.pow(unit,wordAttr.time - 1);
                list3[wordAttr.wordPos] = word;
            }

            String outStr = "" + wordCntForOneFile + "\r\n";
//            System.out.println(wordCntForOneFile);
            for (int i = 0;i < list1.length;++i) {
//                System.out.print(list1[i] + " ");
                outStr += list2[i] + " ";
            }
//            System.out.println();
            outStr += "\r\n";
//            for (int i = 0;i < list3.length;++i) {
////                System.out.print(list1[i] + " ");
//                outStr += list3[i] + " ";
//            }
//            outStr += "\r\n";
            for (int i = 0;i < list2.length;++i) {
//                System.out.print(list2[i] + " ");
                outStr += list1[i] + " ";
//                outStr += list3[i] + " " + list2[i] + " ";
            }
//            System.out.println();
            outStr += "\r\n";
            for(int i = 0;i < wordCntForOneFile;++i)
                for(int j = 0;j < wordCntForOneFile;++j){
                    linkWeight[i][j] = linkWeight[i][j] * (float)Math.pow(unit,linkTime[i][j]);
                    if(linkWeight[i][j] > 0){
//                        System.out.print(i + " " + j + " " + link[i][j] + " ");
                        outStr += i + " " + j + " " + linkWeight[i][j] + " ";
                    }
                }
//            System.out.println();
            outStr += "\r\n" + FileUtils.readFileToString(new File(labelPathPrefix+inn+".label")) + "\r\n";
//            for(int i = 0;i < wordCntForOneFile;++i) {
//                for (int j = 0; j < wordCntForOneFile; ++j) {
//                    outStr += link[i][j] + " ";
//                }
//                outStr += "\r\n";
//            }
//            outStr += "\r\n";

            try {
                FileUtils.writeStringToFile(outFile , outStr);
            } catch (IOException e) {
                e.printStackTrace();
            }
//            System.exit(-1);
//            System.out.println(filesCnt);
            filesCnt++;
            if(filesCnt % 500 == 0) {
                System.out.println("process " + +inner + ": " + filesCnt + " files completed!");
                long endTime=System.currentTimeMillis();
                System.out.println( (endTime-starTime) / 1000.0 / 60 + " min used!");
            }
//            if(filesCnt > 0){
//                break;
//            }
        }

        //word2vec
//        bw.close();
//        writer.close();

        long endTime=System.currentTimeMillis();
        System.out.println("process " + inner + " finally use" + (endTime-starTime) / 1000.0 / 60 + "minutes!");
    }

    private String processFile(File file) throws IOException {
        List<String> lines = FileUtils.readLines(file);
        String result = "",tmp;
        int flag = 0;
        for(int i = 0;i < lines.size();++i){
            tmp = lines.get(i);
            if(tmp.contains("</text>"))break;
            if(flag == 1 || tmp.contains("<headline>")) {
                if(!tmp.contains("."))
                    tmp+=".";
                result += tmp;
            }
            if(tmp.contains("<text>"))
                flag = 1;
        }
        return result;
    }

    private String processNYT(File file) throws IOException {
        List<String> lines = FileUtils.readLines(file);
        String result = "",tmp;
        int flag = 0;
        for(int i = 0;i < lines.size();++i){
            tmp = lines.get(i).trim();
            if(tmp.equals("</body.content>"))break;
            if(flag == 1) {
                int x = tmp.indexOf('>')+1,y = tmp.lastIndexOf('<');
                if(x < y && y - x < 10000) {
//                    System.out.println(y-x+""+"!!!!!!!!!!!!!\n\n\n");
                    tmp = tmp.substring(x, y);
                    if (!tmp.endsWith("."))
                        tmp += ".";
                    result += tmp + "\r\n";
                }
            }
            if(tmp.contains("<hedline>") || tmp.contains("<body.content>"))
                flag = 1;
            else if(tmp.equals("</hedline>"))
                flag = 0;
        }
        return result;
    }

    private void initEdgeWeightMap(HashMap<String, Float> edgeWeightMap) {
        edgeWeightMap.put("agent",2.5f);edgeWeightMap.put("appos",2.1f);edgeWeightMap.put("nmod:appos",2.01f);
        edgeWeightMap.put("nmod:poss",2.21f);edgeWeightMap.put("nmod",2.39f);edgeWeightMap.put("conj",2.38f);
        edgeWeightMap.put("ccomp",1.31f);edgeWeightMap.put("xcomp",1.4f);edgeWeightMap.put("amod",1.38f);
        edgeWeightMap.put("csubj",1.62f);edgeWeightMap.put("csubjpass",1.6f);edgeWeightMap.put("iobj",1.51f);
        edgeWeightMap.put("dobj",1.52f);edgeWeightMap.put("nsubj",1.9f);
        edgeWeightMap.put("nsubjpass",1.88f);edgeWeightMap.put("compound",3.29f);edgeWeightMap.put("acl:relcl",1.41f);
        edgeWeightMap.put("acl",1.43f);edgeWeightMap.put("advcl",1.32f);edgeWeightMap.put("dep",1.29f);
    }

    private void printVectors(String outPath) throws IOException {
        FileWriter writer = new FileWriter(outPath);
        BufferedWriter bw = new BufferedWriter(writer);

        Set<Map.Entry<String, Integer>> entryseSet = printWord2Index.entrySet();
        int entryseSetLength = entryseSet.size();

        System.out.println("words total: " + entryseSetLength);
        String[] stringList = new String[entryseSetLength];
        for (Map.Entry<String, Integer> entry:entryseSet) {
            stringList[entry.getValue()] = entry.getKey();
        }
        for(int i = 0;i < entryseSetLength;++i){
//            System.out.println(stringList[i]);
            float[] vector = vec.getWordVector(stringList[i]);
            if(vector == null){
                int len = 50;
                Random random = new Random();
                for (int j = 0; j < len; j++) {
                    bw.write( ((random.nextDouble() - 0.5) / len)  + " " );
                }
            }else {
                for (int k = 0; k < vector.length; ++k)
                    bw.write(vector[k] + " ");
            }
            bw.write("\r\n");
        }
        bw.close();
        writer.close();
    }

    private void printVecFirst(String outPath) throws IOException {
        FileWriter writer = new FileWriter(outPath);
        BufferedWriter bw = new BufferedWriter(writer);
        for(String str:vec.getWordMap().keySet()){
            float[] vector = vec.getWordVector(str);
            for (int k = 0; k < vector.length; ++k)
                bw.write(vector[k] + " ");
            bw.write("\r\n");
        }
        bw.close();
        writer.close();
    }

    private void printWord2IndexFill() {
        int wordCnt = 0;
        for(String str:vec.getWordMap().keySet()){
            printWord2Index.put(str,wordCnt++);
        }
    }

    private class WordAttr {
        int wordPos;
        float weight;
        int time;
        WordAttr(int wordPos, float weight){
            this.wordPos = wordPos;
            this.weight = weight;
            this.time = 1;
        }
    }

    private class edgeAttr {
        float weight;
        int time;
        edgeAttr(){
            this.weight = 0;
            this.time = 0;
        }
    }
}
