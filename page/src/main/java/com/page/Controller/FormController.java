package com.page.Controller;

import org.springframework.stereotype.Controller;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RequestPart;
import org.springframework.web.multipart.MultipartFile;
import org.springframework.web.client.RestTemplate;
import org.springframework.ui.Model;
import org.springframework.context.annotation.Configuration;
import org.springframework.context.annotation.Bean;
import org.springframework.http.HttpEntity;
import org.springframework.http.HttpHeaders;
import org.springframework.http.MediaType;
import org.springframework.http.ResponseEntity;
import org.springframework.util.LinkedMultiValueMap;
import org.springframework.util.MultiValueMap;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;

import java.text.SimpleDateFormat;
import java.util.Date;

@Controller
public class FormController {

    @PostMapping("/upload")
    public String handleFileUpload(@RequestParam(value="image", required=false) MultipartFile multipartFile, Model model) throws IOException {
        String flaskAppUrl = "http://localhost:5000/upload"; // 플라스크 서버 URL

        // 플라스크 서버로 전송할 요청 헤더 설정
        HttpHeaders headers = new HttpHeaders();
        headers.setContentType(MediaType.MULTIPART_FORM_DATA);

        // MultipartFile을 File로 변환
        File convertedFile = convertMultipartFileToFile(multipartFile);

        MultiValueMap<String, Object> body = new LinkedMultiValueMap<>();
        body.add("image", convertedFile);


        // 플라스크 서버에 전송할 요청 객체 생성
        HttpEntity<MultiValueMap<String, Object>> requestEntity = new HttpEntity<>(body, headers);

        // 플라스크 서버로 POST 요청 보내기
        RestTemplate restTemplate = new RestTemplate();
        ResponseEntity<byte[]> responseEntity = restTemplate.postForEntity(flaskAppUrl, requestEntity, byte[].class);

        // 플라스크 서버로부터의 응답을 파일로 저장
        File receivedImageFile = saveReceivedImage(responseEntity.getBody());

        // 모델에 전달할 파일 경로 설정
        model.addAttribute("imagePath", receivedImageFile.getAbsolutePath());

        return "uploadsuccess";
    }
    private File convertMultipartFileToFile(MultipartFile File) throws IOException {
        // 현재 시간 정보를 활용하여 파일명 생성
        String timestamp = new SimpleDateFormat("yyyyMMdd_HHmmss").format(new Date());
        String originalFileName = File.getOriginalFilename();
        String newFileName = timestamp + "_" + originalFileName;

        // File 객체 생성 시 동적으로 생성한 파일명 사용
        File convertedFile = new File(newFileName);
        try (FileOutputStream fos = new FileOutputStream(convertedFile)) {
            fos.write(File.getBytes());
        }
        return convertedFile;
    }

    private File saveReceivedImage(byte[] imageBytes) throws IOException {
        File receivedImageFile = new File("received_image.jpg");
        try (FileOutputStream fos = new FileOutputStream(receivedImageFile)) {
            fos.write(imageBytes);
        }
        return receivedImageFile;
    }
}