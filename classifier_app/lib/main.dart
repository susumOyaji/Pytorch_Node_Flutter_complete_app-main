import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import 'package:image_picker/image_picker.dart';
import 'dart:io';
import 'dart:convert';
import 'package:http/http.dart' as http;

void main() => runApp(ChangeNotifierProvider(
      create: (context) => ChangeImageState(),
      child: const MyApp(),
    ));

class MyApp extends StatelessWidget {
  const MyApp({Key? key}) : super(key: key);
  // This widget is the root of your application.
  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      // Application name
      title: 'Flutter Hello World',
      // Application theme data, you can set the colors for the application as
      // you want
      theme: ThemeData(
        primarySwatch: Colors.blue,
      ),
      // A widget which will be started on application startup
      home: const SafeArea(child: MyHomePage(title: 'Flutter Demo Home Page')),
    );
  }
}

class MyHomePage extends StatelessWidget {
  final String title;
  const MyHomePage({Key? key, required this.title}) : super(key: key);

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: Column(
        mainAxisAlignment: MainAxisAlignment.spaceEvenly,
        children: const <Widget>[
          Expanded(
            flex: 2,
            child: ShowPicture(),
          ),
          Expanded(
            flex: 1,
            child: ButtonAction(),
          ),
        ],
      ),
    );
  }
}

class ChangeImageState with ChangeNotifier {
  bool isCat = true;
  late File _image;
  bool loaded = false;
  void saveImage(image) async {
    _image = File(image);
    var response = await getPrediction(_image);
    isCat = "Cat" == response.body;
    loaded = true;
    notifyListeners();
  }

  void toggleCatState() {
    isCat = !isCat;
    notifyListeners();
  }
}

Future<http.Response> getPrediction(File image) {
  String base64Image = base64Encode(image.readAsBytesSync());
  return http.post(
    Uri.parse('http://192.168.205.85:8085/predict'),
    body: {
      "image": base64Image,
    },
  );
}

class ShowPicture extends StatefulWidget {
  const ShowPicture({Key? key}) : super(key: key);
  @override
  _ShowPictureState createState() => _ShowPictureState();
}

class _ShowPictureState extends State<ShowPicture> {
  @override
  initState() {
    super.initState();
  }

  @override
  Widget build(BuildContext context) {
    return Consumer<ChangeImageState>(
      builder: (context, data, child) => SizedBox(
        height: MediaQuery.of(context).size.width * 0.7,
        child: data.loaded
            ? Image.file(data._image)
            : Image.asset('images/fun-dog.jpg'),
      ),
    );
  }
}

class ButtonAction extends StatefulWidget {
  const ButtonAction({Key? key}) : super(key: key);
  @override
  _ButtonActionState createState() => _ButtonActionState();
}

class _ButtonActionState extends State<ButtonAction> {
  @override
  initState() {
    super.initState();
  }

  final _picker = ImagePicker();
  // Implementing the image picker
  Future<void> _openImagePicker() async {
    final XFile? pickedImage =
        await _picker.pickImage(source: ImageSource.gallery);
    if (pickedImage != null) {
      setState(() {
        var outState = context.read<ChangeImageState>();
        outState.saveImage(pickedImage.path);
      });
    }
  }

  @override
  Widget build(BuildContext context) {
    final ButtonStyle style = ElevatedButton.styleFrom(
        // primary: Colors.black,
        textStyle: const TextStyle(
      fontSize: 18,
      fontWeight: FontWeight.w300,
    ));
    return Stack(
      clipBehavior: Clip.none,
      children: [
        Positioned(
          top: -50,
          child: Container(
            decoration: const BoxDecoration(
                // color: const Color(0xff7c94b6),
                color: Colors.black,
                borderRadius: BorderRadius.only(
                  topLeft: Radius.circular(15),
                  topRight: Radius.circular(15),
                )),
            height: 315,
            width: MediaQuery.of(context).size.width,
            child: Center(
              child: Row(
                mainAxisAlignment: MainAxisAlignment.spaceAround,
                children: <Widget>[
                  SizedBox(
                      height: 140,
                      width: 70,
                      child: Column(
                        children: [
                          Image.asset('images/dog.png'),
                          Padding(
                            padding: const EdgeInsets.all(8.0),
                            child: Consumer<ChangeImageState>(
                              builder: (context, data, child) => Icon(
                                data.isCat ? Icons.thumb_down : Icons.thumb_up,
                                color:
                                    data.isCat ? Colors.white : Colors.yellow,
                                size: 40.0,
                              ),
                            ),
                          )
                        ],
                      )),
                  SizedBox(
                    height: 60,
                    width: MediaQuery.of(context).size.width * 0.3,
                    child: ElevatedButton(
                      style: style,
                      onPressed: () {
                        // var toggle = context.read<ChangeImageState>();
                        // toggle.toggleCatState();
                        _openImagePicker();
                      },
                      child: Row(
                        mainAxisAlignment: MainAxisAlignment.center,
                        children: [
                          IconButton(
                            icon: const Icon(Icons.image_search),
                            // tooltip: 'Load image from gallery',
                            onPressed: () {
                              var toggle = context.read<ChangeImageState>();
                              _openImagePicker();
                            },
                          ),
                          // Consumer<ChangeImageState>(
                          //   builder: (context, data, child) => Text(
                          //     '${data.isCat}',
                          //     style: const TextStyle(
                          //       color: Colors.white,
                          //     ),
                          //   ),
                          // ),
                        ],
                      ),
                    ),
                  ),
                  SizedBox(
                    height: 140,
                    width: 70,
                    child: Column(
                      children: [
                        Image.asset('images/cat.png'),
                        Padding(
                          padding: const EdgeInsets.all(8.0),
                          child: Consumer<ChangeImageState>(
                            builder: (context, data, child) => Icon(
                              !data.isCat ? Icons.thumb_down : Icons.thumb_up,
                              color: !data.isCat ? Colors.white : Colors.yellow,
                              size: 40.0,
                            ),
                          ),
                        )
                      ],
                    ),
                  )
                ],
              ),
            ),
          ),
        ),
      ],
    );
  }
}
